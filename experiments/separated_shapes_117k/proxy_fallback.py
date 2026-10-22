"""Objective-only acceptance for a bounded P-Flow rescue proposal."""
from dataclasses import dataclass
import math

import torch

from p_flow.armijo import Objective
from p_flow.p_flow import project_source


@dataclass
class ProxyTrial:
    source: torch.Tensor
    accepted: bool
    value: float
    margin: float
    baseline_values: list[float]
    trials: list[dict[str, float | bool]]


@torch.no_grad()
def proxy_step(source: torch.Tensor, gradient: torch.Tensor, objective: Objective,
               cap: float, max_trials: int, project: bool) -> ProxyTrial:
    """Try the cap then halve; require repeatable decrease beyond a roundoff guard.

    No proxy slope enters acceptance. The guard is the larger of four times
    observed baseline variation and eight source-dtype epsilons times the loss
    scale. It is a declared numerical safeguard, not a statistical noise bound.
    """
    baseline = [objective(source) for _ in range(3)]
    if not all(map(math.isfinite, baseline)):
        raise FloatingPointError("Nonfinite baseline objective")
    margin = max(4*(max(baseline)-min(baseline)),
                 8*torch.finfo(source.dtype).eps*max(1., max(map(abs, baseline))))
    norm = float(gradient.norm())
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite proxy gradient")
    result = ProxyTrial(source.detach().clone(), False, baseline[-1], margin, baseline, [])
    if norm == 0:
        return result
    direction = -gradient/norm
    for index in range(max_trials):
        displacement = cap*2.**-index
        candidate = source+displacement*direction
        if project:
            candidate = project_source(candidate)
        actual_displacement = float((candidate-source).norm())
        if torch.equal(candidate, source):
            break
        if actual_displacement > cap*(1+1e-6):
            continue
        value = objective(candidate)
        record = dict(displacement=actual_displacement, value=value, accepted=False)
        result.trials.append(record)
        if math.isfinite(value) and value < min(baseline)-margin:
            repeated = objective(candidate)
            record["repeat_value"] = repeated
            if math.isfinite(repeated) and repeated < min(baseline)-margin:
                record["accepted"] = True
                return ProxyTrial(candidate.detach().clone(), True, repeated, margin, baseline, result.trials)
    objective(source)  # Restore the evaluator cache after rejected proposals.
    return result
