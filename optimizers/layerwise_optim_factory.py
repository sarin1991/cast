import json
from typing import Dict, Any, Type, Optional
import torch

# Register whatever implementations you want to swap between
from .layerwise_low_rank_ns_muon import (
    LayerwiseLowRankNSMuonOptimizer
)
from .layerwise_all_low_rank_muon import (
    LayerwiseAllLowRankMuonOptimizer
)
from .layerwise_low_rank_muon import (
    LayerwiseLowRankMuonOptimizer
)
from .layerwise_muon import (
    LayerwiseMuonOptimizer
)
from .layerwise_adam import (
    LayerwiseAdamOptimizer
)

LAYERWISE_OPT_REGISTRY: Dict[str, Type[torch.optim.Optimizer]] = {
    "layerwise_low_rank_ns_muon": LayerwiseLowRankNSMuonOptimizer,
    "layerwise_all_low_rank_muon": LayerwiseAllLowRankMuonOptimizer,
    "layerwise_low_rank_muon": LayerwiseLowRankMuonOptimizer,
    "layerwise_muon": LayerwiseMuonOptimizer,
    "layerwise_adam": LayerwiseAdamOptimizer,
    # "my_other_opt": MyOtherHookOptimizer,
}

def create_layerwise_optimizer(
    name: str,
    param_groups,
    lr: float,
    kwargs_json: str = "{}",
):
    """
    Creates an optimizer instance from a name + JSON kwargs.

    name: key in HOOK_OPT_REGISTRY, or "none"/None to disable.
    param_groups: exactly what you'd pass to the optimizer (e.g. [{'params': ...}])
    lr: base learning rate
    kwargs_json: JSON string of extra kwargs for the optimizer ctor
    """
    if name is None or name == "" or name == "none":
        raise ValueError("layerwise_optim was 'none'")

    if name not in LAYERWISE_OPT_REGISTRY:
        raise ValueError(
            f"Unknown hook optimizer '{name}'. "
            f"Valid: {list(LAYERWISE_OPT_REGISTRY.keys()) + ['none']}"
        )

    try:
        extra: Dict[str, Any] = json.loads(kwargs_json or "{}")
        if not isinstance(extra, dict):
            raise ValueError("kwargs_json must decode to a JSON object (dict).")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse hook optimizer kwargs JSON: {e}") from e

    opt_cls = LAYERWISE_OPT_REGISTRY[name]
    return opt_cls(param_groups, lr=lr, **extra)