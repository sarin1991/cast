import torch


class LayerwiseAdamOptimizer(torch.optim.Adam):
    """
    Layerwise Adam wrapper that lets you supply externally-computed grads
    and then run a standard torch.optim.Adam step.
    """
    def __init__(self, param_groups, **kwargs):
        super().__init__(param_groups, **kwargs)

    @torch.no_grad()
    def step_from_grads(self, params, grads, closure=None, force_sync_on_none: bool = False):
        """
        params: iterable of nn.Parameter managed by this optimizer
        grads: iterable of gradient tensors (same order/length), may contain None

        Assigns provided grads to p.grad, then calls the usual Adam step().

        force_sync_on_none:
            If True, when g is None set p.grad = zeros_like(p) (optionally useful in DDP/FSDP
            scenarios to force sync behavior).
        """
        for p, g in zip(params, grads):
            if g is None:
                p.grad = torch.zeros_like(p) if force_sync_on_none else None
            else:
                p.grad = g.detach()
        return super().step(closure=closure)