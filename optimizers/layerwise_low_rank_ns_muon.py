import torch


@torch.no_grad()
def decompress(P: torch.Tensor, X_lr: torch.Tensor):
    """
    Decompress low-rank tensor stored as:
        X = P @ X_lr
    where:
        P: (m, r) with orthonormal columns (basis)
        X_lr: (r, n) low rank tensor
    """
    X = P @ X_lr
    return X


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


class LowRankNSMuonOptimizer(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, momentum_rank=128):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, momentum_rank=momentum_rank)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                gradient = p.grad.detach().to(torch.bfloat16) # cast in case fp32
                if p.dtype == torch.float32:
                    isfloat32 = True
                else:
                    isfloat32 = False
                if len(state) == 0:
                    r = group["momentum_rank"]
                    projection_matrix_init = torch.randn((gradient.shape[0], r), device=p.device, dtype=torch.bfloat16)
                    state["projection_matrix"] = zeropower_via_newtonschulz5(projection_matrix_init, steps=5)
                    momentum_buffer_low_rank_init = torch.randn((r, gradient.shape[1]), device=p.device, dtype=torch.bfloat16)
                    state["momentum_buffer_low_rank"] = momentum_buffer_low_rank_init
                    momentum = torch.zeros_like(p,dtype=torch.bfloat16)
                else:
                    momentum = decompress(state["projection_matrix"], state["momentum_buffer_low_rank"])
                decay = (1 - group["lr"] * group["weight_decay"])
                beta = group["momentum"]
                # build weight
                if isfloat32:
                    W = p
                else:
                    W = p.float() 
                # Update momentum
                momentum.lerp_(gradient, 1 - beta)
                update = gradient.lerp_(momentum, beta)
                # update projection matrix and momentum low rank
                q = zeropower_via_newtonschulz5(state["momentum_buffer_low_rank"].T, steps=5)
                state["projection_matrix"].copy_(zeropower_via_newtonschulz5(momentum @ q, steps=5))
                torch.matmul(state["projection_matrix"].T, momentum, out=state["momentum_buffer_low_rank"])
                # remove eigen vals for update
                update_q = zeropower_via_newtonschulz5(state["projection_matrix"].T @ update, steps=5)
                s = max(1, update.size(-2) / update.size(-1))**0.5
                update.addmm_(state["projection_matrix"], update_q, beta=0.0, alpha=s)
                W.add_(update.reshape(W.shape), alpha=-group["lr"])
                if not isfloat32:
                    copy_stochastic_(p, W)
        return loss

class LayerwiseLowRankNSMuonOptimizer(LowRankNSMuonOptimizer):
    """
    Shell for a layerwise Muon optimizer.
    Fill in the logic in future work.
    """
    def __init__(self, param_groups, **kwargs):
        super().__init__(param_groups, **kwargs)
    
    @torch.no_grad()
    def step_from_grads(self, params, grads):
        """
        params: iterable of nn.Parameter managed by this optimizer
        grads: iterable of gradient tensors (same order/length), may contain None

        Assigns provided grads to p.grad, then applies the normal Muon step.
        """
        for p, g in zip(params, grads):
            if g is None:
                p.grad = None
            else:
                p.grad = g.detach()
        out = self.step()
        for p in params:
            p.grad = None
        return out
