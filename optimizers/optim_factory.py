import json
from typing import Dict, Any, Type, Optional
import torch

# Register whatever implementations you want to swap between
from .layerwise_low_rank_pe_wres_muon import (
    LayerwiseLowRankPEWRESMuonOptimizer, LowRankPEWRESMuonOptimizer
)
from .layerwise_low_rank_pe_muon import (
    LayerwiseLowRankPEMuonOptimizer, LowRankPEMuonOptimizer
)
from .layerwise_low_rank_ns_muon import (
    LayerwiseLowRankNSMuonOptimizer, LowRankNSMuonOptimizer
)
from .layerwise_all_low_rank_muon import (
    LayerwiseAllLowRankMuonOptimizer, AllLowRankMuonOptimizer
)
from .layerwise_low_rank_muon import (
    LayerwiseLowRankMuonOptimizer, LowRankMuonOptimizer
)
from .layerwise_muon import (
    LayerwiseMuonOptimizer, MuonOptimizer
)
from .layerwise_adam import (
    LayerwiseAdamOptimizer
)

LAYERWISE_OPT_REGISTRY: Dict[str, Type[torch.optim.Optimizer]] = {
    "layerwise_low_rank_pe_wres_muon": LayerwiseLowRankPEWRESMuonOptimizer,
    "layerwise_low_rank_pe_muon": LayerwiseLowRankPEMuonOptimizer,
    "layerwise_low_rank_ns_muon": LayerwiseLowRankNSMuonOptimizer,
    "layerwise_all_low_rank_muon": LayerwiseAllLowRankMuonOptimizer,
    "layerwise_low_rank_muon": LayerwiseLowRankMuonOptimizer,
    "layerwise_muon": LayerwiseMuonOptimizer,
    "layerwise_adam": LayerwiseAdamOptimizer,
    "low_rank_pe_wres_muon": LowRankPEWRESMuonOptimizer,
    "low_rank_pe_muon": LowRankPEMuonOptimizer,
    "low_rank_ns_muon": LowRankNSMuonOptimizer,
    "all_low_rank_muon": AllLowRankMuonOptimizer,
    "low_rank_muon": LowRankMuonOptimizer,
    "muon": MuonOptimizer,
    "adam": torch.optim.Adam,
    # "my_other_opt": MyOtherHookOptimizer,
}

def create_optimizer(
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