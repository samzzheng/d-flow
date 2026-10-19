"""Is bf16 viable for D-Flow / DLR-D-Flow, without retraining? Quick local test.

The model was trained under torch.autocast(dtype=torch.bfloat16) (see
autocast_context in train_fm_unet.py), so bf16 inference is the regime the
weights already know, not something new. This checks whether the two
precision-sensitive downstream computations still work under it:

  D-Flow    exact source gradient through the 100-step Euler flow, fp32/TF32
            vs bf16 autocast, against a float64 reference.
  DLR       the hand-written JVP/VJP tangent pass (dlr_dflow.tangent_unet),
            fp32 vs bf16, against the same float64 reference -- and the full
            KSL evolution's terminal transpose action against autograd, at
            the starting source and a near-floor point (the regime found
            sensitive to small errors: cosine +-1, large relative error).

64px, 1-2 samples, on a free GPU. Usage: python scripts/test_bf16_precision.py [cuda:0]
"""
import json, math, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import test_parallel_bug_integrator as H
import dlr_dflow.dlr_dflow as dd
import dlr_dflow.tangent_unet as tangent_mod
import models.unet.default as default_mod
from dlr_dflow import enriched as E
from dlr_dflow.tangent_unet import unet_jvp
from dlr_dflow.graphed_products import GraphedAProducts

# get_timestep_embedding (models/unet/unet.py) hardcodes dtype=torch.float32
# internally for the sinusoidal computation and never casts back to the
# caller's dtype -- fine under fp32/TF32, but breaks a genuinely-bf16 model
# (its time_proj Linear expects bf16 input). Patched here at the call sites
# used by the hand JVP/VJP pass and the ordinary forward, rather than in the
# shared model code: this is a test-only bf16 viability check.
_orig_get_timestep_embedding = tangent_mod.get_timestep_embedding
def _dtype_matched_embedding(timesteps, *a, **kw):
    return _orig_get_timestep_embedding(timesteps, *a, **kw).to(timesteps.dtype)
tangent_mod.get_timestep_embedding = _dtype_matched_embedding
default_mod.get_timestep_embedding = _dtype_matched_embedding

# torch.linalg.qr/svd have no CUDA kernel for BFloat16 (geqrf_cuda is fp32/
# fp64/complex only) -- this is a library gap, not a precision choice. The
# realistic bf16 DLR scheme is therefore mixed precision: bf16 for the JVP/
# VJP/conv work, fp32 just for the linear-algebra calls the projector-
# splitting step needs every flow step. Patched at the module level used by
# enriched.py, not touched in enriched.py itself.
_orig_qr, _orig_svd = torch.linalg.qr, torch.linalg.svd
def _qr_fp32(a, *ar, **kw):
    if a.dtype == torch.bfloat16:
        q, r = _orig_qr(a.float(), *ar, **kw)
        return q.to(a.dtype), r.to(a.dtype)
    return _orig_qr(a, *ar, **kw)
def _svd_fp32(a, *ar, **kw):
    if a.dtype == torch.bfloat16:
        u, s, vt = _orig_svd(a.float(), *ar, **kw)
        return u.to(a.dtype), s.to(a.dtype), vt.to(a.dtype)
    return _orig_svd(a, *ar, **kw)
# patch in place on the real torch.linalg module so every other function
# (vector_norm, etc.) on it is untouched
torch.linalg.qr = _qr_fp32
torch.linalg.svd = _svd_fp32

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0"); torch.cuda.set_device(dev)
R, STEPS, RANK, M, P = 64, 100, 8, 4, 4
model, tag = H.load(R, dev)
model_bf16 = None  # built lazily below, after model is loaded
products = GraphedAProducts(model, torch.zeros(1, 1, R, R, device=dev))
angles = torch.linspace(0, math.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
out = {}


def to_bf16_model(m):
    """Real cast (not autocast): every weight and every intermediate tensor
    genuinely bf16, so the hand-written aten calls in tangent_unet.py never
    see a mixed-dtype pair (autocast promotes GroupNorm inputs but the hand
    _gn() path calls torch.ops.aten.native_group_norm directly and bypasses
    that promotion -- see fast_groupnorm.py's comment on the same issue)."""
    import copy
    m16 = copy.deepcopy(m).to(torch.bfloat16)
    m16.eval()
    for p_ in m16.parameters():
        p_.requires_grad_(False)
    return m16


# ---- 1. one JVP column at mid-flow, vs a float64 forward-mode reference ----
z = torch.randn(1, 1, R, R, generator=torch.Generator(device=dev).manual_seed(0), device=dev)
with torch.no_grad():
    x = z.clone()
    for i in range(50):
        x = x + 0.01 * dd.velocity(model, i * 0.01, x)
q = torch.randn(4, 1, R, R, generator=torch.Generator(device=dev).manual_seed(1), device=dev)
q = q / q.flatten(1).norm(dim=1).view(-1, 1, 1, 1)

model64 = H.load(R, dev)[0].double()
model64.time_proj.register_forward_pre_hook(lambda m, i: (i[0].double(),))
_, ref = torch.func.vmap(
    lambda qq: torch.func.jvp(lambda X: model64(X, torch.full((1,), 0.5, device=dev, dtype=torch.float64)),
                              (x.double(),), (qq[None],)), out_dims=(None, 0))(q.double())
ref = ref.reshape(4, 1, R, R).float()
model_bf16 = to_bf16_model(model)
with torch.no_grad():
    fp32_out = unet_jvp(model, 0.5, x, q)[1]
    bf16_out = unet_jvp(model_bf16, 0.5, x.to(torch.bfloat16), q.to(torch.bfloat16))[1].float()
err = lambda a: ((a - ref).norm() / ref.norm()).item()
out["jvp_column_error"] = {"fp32_tf32": err(fp32_out), "bf16_autocast": err(bf16_out)}
print("JVP column relative error vs float64 reference:")
for k, v in out["jvp_column_error"].items(): print(f"  {k:14s} {v:.3e}")

# ---- 2. exact D-Flow gradient and DLR evolution, fp32 vs bf16 -------------
def exact_grad(z, y, use_bf16):
    m = model_bf16 if use_bf16 else model
    zz = (z.to(torch.bfloat16) if use_bf16 else z).detach().clone().requires_grad_(True)
    xt = dd.euler_flow(m, zz, STEPS)
    obj, rel = H.bench.data_objective(H.bench.sparse_radon(xt.float(), angles), y, R, "relative_scaled")
    g, = torch.autograd.grad(obj, zz)
    return g.flatten().float(), math.sqrt(rel.item())


def dlr_grad(z, y, use_bf16):
    # products (CUDA-graph-captured A/AT) are fp32-only; bf16 evolution runs
    # eager without them via a fresh, ungraphed A/AT pair for this model dtype.
    m = model_bf16 if use_bf16 else model
    zk = z.to(torch.bfloat16) if use_bf16 else z
    x, factors, a1 = E.evolve_trace_centered_enriched(
        m, zk, STEPS, rank=RANK, trace_probes=M, initialization_oversampling=8,
        enrichment_probes=P, factor_stride=1, products=(None if use_bf16 else products),
        zero_terminal_baseline=False)
    xt = x.float().detach().requires_grad_(True)
    obj, _ = H.bench.data_objective(H.bench.sparse_radon(xt, angles), y, R, "relative_scaled")
    g, = torch.autograd.grad(obj, xt)
    return g.flatten(), factors, float(a1)


points = {}
for sample in (0, 1):
    z0, y = H.make_problem(R, sample, data, angles, dev)
    pts = H.exact_iterates(model, z0, y, R, angles, [10])
    points[(sample, 0)] = (z0, y)
    points[(sample, 10)] = (pts[10], y)

rows = []
for (sample, k), (zk, y) in points.items():
    g32, rel = exact_grad(zk, y, False)
    torch.cuda.synchronize(); t0 = time.time()
    g16, _ = exact_grad(zk, y, True)
    torch.cuda.synchronize(); t_dflow_bf16 = time.time() - t0
    torch.cuda.synchronize(); t0 = time.time()
    _ = exact_grad(zk, y, False)
    torch.cuda.synchronize(); t_dflow_fp32 = time.time() - t0
    dflow_cos = (g16 @ g32 / (g16.norm() * g32.norm())).item()
    dflow_err = ((g16 - g32).norm() / g32.norm()).item()

    _, f32, a32 = dlr_grad(zk, y, False)
    torch.cuda.synchronize(); t0 = time.time()
    gx16, f16, a16 = dlr_grad(zk, y, True)
    torch.cuda.synchronize(); t_dlr_bf16 = time.time() - t0
    torch.cuda.synchronize(); t0 = time.time()
    gx32, _, _ = dlr_grad(zk, y, False)
    torch.cuda.synchronize(); t_dlr_fp32 = time.time() - t0
    est32 = a32 * gx32 + f32[2] @ (f32[1].T @ (f32[0].T @ gx32))
    est16 = a16 * gx16 + f16[2].float() @ (f16[1].float().T @ (f16[0].float().T @ gx16.float()))
    dlr_vs_exact32 = ((est32 - g32).norm() / g32.norm()).item()
    dlr_vs_exact16 = ((est16 - g32).norm() / g32.norm()).item()

    row = dict(sample=sample, iterate=k, residual=rel,
               armijo_initial_step_policy=H.bench.ArmijoLBFGS.initial_step_policy,
               armijo_progress_policy=H.bench.ArmijoLBFGS.progress_policy,
               dflow_bf16_vs_fp32_err=dflow_err, dflow_bf16_vs_fp32_cos=dflow_cos,
               dflow_fp32_s=t_dflow_fp32, dflow_bf16_s=t_dflow_bf16,
               dlr_action_err_fp32=dlr_vs_exact32, dlr_action_err_bf16=dlr_vs_exact16,
               dlr_fp32_s=t_dlr_fp32, dlr_bf16_s=t_dlr_bf16)
    rows.append(row)
    print(f"\nsample {sample} it {k}: residual {100*rel:.1f}%")
    print(f"  D-Flow grad: bf16 vs fp32 err {dflow_err:.4f} cos {dflow_cos:+.5f}  "
          f"time fp32 {t_dflow_fp32:.2f}s bf16 {t_dflow_bf16:.2f}s ({t_dflow_fp32/t_dflow_bf16:.2f}x)")
    print(f"  DLR action error vs exact: fp32 {dlr_vs_exact32:.4f}  bf16 {dlr_vs_exact16:.4f}  "
          f"time fp32 {t_dlr_fp32:.2f}s bf16 {t_dlr_bf16:.2f}s ({t_dlr_fp32/t_dlr_bf16:.2f}x)")
    out["points"] = rows
    (ROOT / "outputs/parallel_bug_test/bf16_test.json").write_text(json.dumps(out, indent=1))
print("\nBF16 TEST DONE")
