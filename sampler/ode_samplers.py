from typing import Callable, Dict, List, Optional

import torch


def _euler_step(ode_func: Callable, t: torch.Tensor, x: torch.Tensor, dt: float) -> torch.Tensor:
    return x + dt * ode_func(t, x)


def _rk4_step(ode_func: Callable, t: torch.Tensor, x: torch.Tensor, dt: float) -> torch.Tensor:
    dt_t = x.new_tensor(dt)
    half = x.new_tensor(0.5)
    k1 = ode_func(t, x)
    k2 = ode_func(t + half * dt_t, x + half * dt_t * k1)
    k3 = ode_func(t + half * dt_t, x + half * dt_t * k2)
    k4 = ode_func(t + dt_t, x + dt_t * k3)
    return x + (dt_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _integrate_segment(
    ode_func: Callable,
    x0: torch.Tensor,
    t0: float,
    t1: float,
    method: str,
    step_size: float,
) -> torch.Tensor:
    if step_size <= 0:
        raise ValueError("step_size must be > 0 for fixed-step samplers.")

    total = t1 - t0
    if total == 0:
        return x0

    sign = 1.0 if total > 0 else -1.0
    dt = sign * abs(step_size)
    n_steps = int(abs(total) // abs(step_size))
    remainder = total - n_steps * dt

    x = x0
    t = x0.new_tensor(t0)

    step_fn = _euler_step if method == "euler" else _rk4_step
    for _ in range(n_steps):
        x = step_fn(ode_func, t, x, dt)
        t = t + x0.new_tensor(dt)

    if abs(remainder) > 0:
        x = step_fn(ode_func, t, x, remainder)

    return x


def ode_integrate(
    ode_func: Callable,
    init_x: torch.Tensor,
    ode_opts: Optional[Dict] = None,
    t_eps: float = 0.0,
    init_t: float = 0.0,
    final_t: float = 1.0,
    t_arr: Optional[List[float]] = None,
    intermediate_points: bool = False,
    sampler: str = "torchdiffeq",
) -> torch.Tensor:
    """Integrate dx/dt = f(t, x) with selectable sampler.

    sampler:
    - "torchdiffeq": delegates to torchdiffeq.odeint
    - "euler": fixed-step explicit Euler
    - "rk4": fixed-step Runge-Kutta 4
    """
    ode_opts = {} if ode_opts is None else dict(ode_opts)

    if t_arr is None:
        times = [init_t - t_eps, final_t]
    else:
        times = list(t_arr)
        if len(times) < 2:
            raise ValueError("t_arr must include at least two time points.")

    if sampler == "torchdiffeq":
        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise ImportError("sampler='torchdiffeq' requires torchdiffeq to be installed.") from exc
        t = torch.tensor(times, dtype=init_x.dtype, device=init_x.device)
        z = odeint(ode_func, init_x, t, **{"atol": 1e-5, "rtol": 1e-5, **ode_opts})
        return z if intermediate_points else z[-1]

    if sampler not in {"euler", "rk4"}:
        raise ValueError(f"Unknown sampler '{sampler}'. Use one of: torchdiffeq, euler, rk4.")

    step_size = ode_opts.get("options", {}).get("step_size", 0.02)
    x = init_x
    traj = [x]
    for t0, t1 in zip(times[:-1], times[1:]):
        x = _integrate_segment(
            ode_func=ode_func,
            x0=x,
            t0=float(t0),
            t1=float(t1),
            method=sampler,
            step_size=float(step_size),
        )
        traj.append(x)

    if not intermediate_points:
        return traj[-1]
    return torch.stack(traj, dim=0)
