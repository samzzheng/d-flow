import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

from sampler.ode_samplers import ode_integrate


def get_init_x(
    x_dims: torch.Size,
    ode_func: Callable,
    y: torch.Tensor,
    init_method: Any = "random",
    init_ode_opts: Optional[Dict] = None,
    blend_coef: float = 1.0,
    init_t: float = 0.0,
    final_t: float = 1.0,
    sampler: str = "rk4",
) -> torch.Tensor:
    init_ode_opts = {} if init_ode_opts is None else init_ode_opts
    if init_method == "random":
        return torch.randn(x_dims, device=y.device, dtype=y.dtype)

    if init_method in {"backward", "backward_blend"}:
        with torch.no_grad():
            init_x = ode_integrate(
                ode_func,
                y,
                ode_opts=init_ode_opts,
                init_t=final_t,
                final_t=init_t,
                sampler=sampler,
            )
        if init_method == "backward_blend":
            noise = torch.randn(x_dims, device=y.device, dtype=y.dtype)
            init_x = init_x * (blend_coef**0.5) + noise * ((1.0 - blend_coef) ** 0.5)
        return init_x

    if isinstance(init_method, int):
        generator = torch.Generator(device=y.device)
        generator.manual_seed(init_method)
        return torch.randn(x_dims, generator=generator, device=y.device, dtype=y.dtype)

    raise ValueError(f"Unknown init_method {init_method}")


def dflow(
    ode_func: Callable,
    y: torch.Tensor,
    H: Callable,
    cost_func: Callable,
    x_dims: torch.Size,
    init_method: Any = "random",
    init_ode_opts: Optional[Dict] = None,
    blend_coef: float = 1.0,
    ode_opts: Optional[Dict] = None,
    sampler: str = "rk4",
    optimizer_type: str = "LBFGS",
    lr: float = 1.0,
    optim_steps: int = 30,
    max_iter: int = 20,
    regularizer: Any = None,
    reg_lam: float = 0.0,
    target_cost: Optional[float] = None,
    time_limit: Optional[float] = None,
    t_range: Tuple[float, float] = (0.0, 1.0),
    log_every: int = 10,
) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, List[float]]]:
    init_ode_opts = {} if init_ode_opts is None else init_ode_opts
    ode_opts = {} if ode_opts is None else ode_opts
    start_time = time.time()

    init_t, final_t = t_range
    init_x = get_init_x(
        x_dims=x_dims,
        ode_func=ode_func,
        y=y,
        init_method=init_method,
        init_ode_opts=init_ode_opts,
        blend_coef=blend_coef,
        init_t=init_t,
        final_t=final_t,
        sampler=sampler,
    )
    init_x = init_x.requires_grad_(True)

    if optimizer_type == "LBFGS":
        optimizer = optim.LBFGS([init_x], max_iter=max_iter, lr=lr, line_search_fn="strong_wolfe")
    elif optimizer_type == "SGD":
        optimizer = optim.SGD([init_x], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_type {optimizer_type}")

    metrics: Dict[str, List[float]] = {
        "loss": [],
        "cost": [],
        "reg": [],
        "norm_x0": [],
        "std_x0": [],
        "mean_x0": [],
        "time_sec": [],
    }
    x1_trajectory: List[torch.Tensor] = []
    loss = torch.tensor(0.0, device=init_x.device, dtype=init_x.dtype)

    def closure() -> torch.Tensor:
        nonlocal loss
        optimizer.zero_grad()

        x1 = ode_integrate(
            ode_func,
            init_x,
            ode_opts=ode_opts,
            init_t=init_t,
            final_t=final_t,
            sampler=sampler,
        )

        reg_loss = torch.tensor(0.0, device=init_x.device, dtype=init_x.dtype)
        if regularizer == "chi_d":
            dim_x = torch.tensor(init_x[0].numel(), device=init_x.device, dtype=init_x.dtype)
            init_x_norm = init_x.norm()
            reg_loss = -((dim_x - 1.0) * torch.log(init_x_norm + 1e-8) - init_x_norm.pow(2) / 2.0)

        degraded_x1 = H(x1)
        cost = cost_func(degraded_x1, y)
        loss = cost + reg_lam * reg_loss

        metrics["norm_x0"].append(init_x.norm().item())
        metrics["std_x0"].append(init_x.std().item())
        metrics["mean_x0"].append(init_x.mean().item())
        metrics["cost"].append(cost.item())
        metrics["reg"].append(reg_loss.item())
        metrics["loss"].append(loss.item())
        metrics["time_sec"].append(time.time() - start_time)

        loss.backward()
        return loss

    for step in range(optim_steps):
        if optimizer_type == "LBFGS":
            optimizer.step(closure)
        else:
            _ = closure()
            optimizer.step()

        with torch.no_grad():
            x1 = ode_integrate(
                ode_func,
                init_x,
                ode_opts=ode_opts,
                init_t=init_t,
                final_t=final_t,
                sampler=sampler,
            )
            x1_trajectory.append(x1.detach().cpu())

        if step % max(1, log_every) == 0:
            print(
                f"[Step {step}] loss={metrics['loss'][-1]:.6f} "
                f"cost={metrics['cost'][-1]:.6f} "
                f"time_min={(time.time() - start_time)/60.0:.2f}"
            )

        if target_cost is not None and metrics["cost"][-1] <= target_cost:
            break
        if time_limit is not None and (time.time() - start_time) / 60.0 > time_limit:
            break

    with torch.no_grad():
        x1 = ode_integrate(
            ode_func,
            init_x,
            ode_opts=ode_opts,
            init_t=init_t,
            final_t=final_t,
            sampler=sampler,
        )

    return x1, x1_trajectory, metrics
