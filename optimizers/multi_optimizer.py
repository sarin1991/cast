import torch
from torch.optim import Optimizer

class MultiOptimizer(Optimizer):
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        all_groups = []
        for opt in self.optimizers:
            all_groups.extend(opt.param_groups)
        super().__init__(all_groups, defaults={})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict_list):
        for opt, state in zip(self.optimizers, state_dict_list):
            opt.load_state_dict(state)
