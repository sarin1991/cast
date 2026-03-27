import torch
from .utils import LayerwiseOptimizerMixin

class LayerwiseAdamOptimizer(LayerwiseOptimizerMixin, torch.optim.Adam):
    pass