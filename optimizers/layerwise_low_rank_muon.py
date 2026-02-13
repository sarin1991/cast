import torch

def _orthonormalize_columns(X: torch.Tensor) -> torch.Tensor:
    # reduced QR in fp32 for stability
    Q, _ = torch.linalg.qr(X.to(torch.float32), mode="reduced")
    return Q


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


@torch.no_grad()
def low_rank_projection_block_iteration(P: torch.Tensor, M: torch.Tensor, M_lr: torch.Tensor):
    """
    One warm-started block iteration that updates the low rank projection matrix.

    Inputs:
      P:      (m, r) projection matrix
      M:      (m, n) fp32
      M_lr: (r, n) low rank momentum

    Returns:
      None # inplace updates P & M_lr
    """
    m, n = M.shape
    r, _ = M_lr.shape

    r_buf = torch.empty((r,r),device=P.device,dtype=P.dtype)
    q_buf = torch.empty((n,r),device=P.device,dtype=P.dtype)

    torch.linalg.qr(M_lr.T, mode="reduced",out=(q_buf,r_buf)) # q = (n,r)
    torch.linalg.qr(M @ q_buf, mode="reduced",out=(P,r_buf)) # q = (m,r)
    # Store new low rank momentum
    torch.matmul(P.T, M, out=M_lr)  # (r, n)


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


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update


class LowRankMuonOptimizer(torch.optim.Optimizer):
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
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                gradient = p.grad.detach().float()
                m, n = gradient.shape
                initialized = True
                if len(state) == 0:
                    r = group["momentum_rank"]
                    projection_matrix_init = torch.randn((gradient.shape[0], r), device=p.device, dtype=torch.float32)
                    state["projection_matrix"] = _orthonormalize_columns(projection_matrix_init)
                    momentum_buffer_low_rank_init = torch.randn((r, gradient.shape[1]), device=p.device, dtype=torch.float32)
                    state["momentum_buffer_low_rank"] = momentum_buffer_low_rank_init
                    state["weight_residual"] = torch.zeros_like(momentum_buffer_low_rank_init)
                    initialized = False
                if initialized:
                    momentum = decompress(state["projection_matrix"], state["momentum_buffer_low_rank"])
                else:
                    momentum = torch.zeros_like(p,dtype=torch.float32)
                update = muon_update(gradient, momentum, beta=group["momentum"])
                decay = (1 - group["lr"] * group["weight_decay"])
                W = p.float()
                W.addmm_( state["projection_matrix"], state["weight_residual"], beta=decay, alpha=decay)
                W.add_(update.reshape(W.shape), alpha=-group["lr"])
                p.copy_(W)
                W.sub_(p)
                low_rank_projection_block_iteration(state["projection_matrix"], momentum, state["momentum_buffer_low_rank"])
                torch.matmul(state["projection_matrix"].T, W, out=state["weight_residual"])
        return loss

class LayerwiseLowRankMuonOptimizer(LowRankMuonOptimizer):
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
                continue
            p.grad = g.detach()
        return self.step()
