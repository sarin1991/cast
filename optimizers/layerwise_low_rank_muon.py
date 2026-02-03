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
def low_rank_projection_block_iteration(M: torch.Tensor, M_lr_prev: torch.Tensor):
    """
    One warm-started block iteration that updates the low rank projection matrix.

    Inputs:
      M:      (m, n) fp32
      M_lr_prev: (r, n) previous low rank momentum

    Returns:
      P_new: (m, r) fp32 (new projection matrix)
      M_lr_new: (r, n) fp32 (new low rank momentum)
    """
    m, n = M.shape
    r, _ = M_lr_prev.shape

    Q_prev = _orthonormalize_columns(M_lr_prev.T)  # (n, r)

    # Update P via one subspace iteration step: P <- orth(M Q_prev)
    P_new = _orthonormalize_columns(M @ Q_prev)  # (m, r)

    # Store new low rank momentum
    M_lr_new = P_new.T @ M  # (r, n)

    return P_new, M_lr_new


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


@torch.no_grad()
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """
    Computes Muon update and returns (update, momentum_new).

    Requirements (per your request):
      - No in-place ops for `update` (and we also avoid in-place ops on `grad`).
      - `momentum` is NOT modified in-place either; we return a new momentum tensor.

    grad:     tensor shaped like parameter (2D here)
    momentum: tensor shaped like parameter (2D here)
    """
    # momentum_new = beta*momentum + (1-beta)*grad
    momentum_new = momentum * beta + grad * (1.0 - beta)

    # Nesterov-style blend: update_pre = (1-beta)*grad + beta*momentum_new
    update_pre = grad * (1.0 - beta) + momentum_new * beta if nesterov else momentum_new

    update = zeropower_via_newtonschulz5(update_pre, steps=ns_steps)
    update = update * (max(1.0, update.size(-2) / update.size(-1)) ** 0.5)

    return update, momentum_new


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
                gradient = p.grad.detach()
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
                projection_matrix = state["projection_matrix"]
                momentum_low_rank = state["momentum_buffer_low_rank"]
                if initialized:
                    momentum = decompress(projection_matrix, momentum_low_rank)
                else:
                    momentum = torch.zeros_like(p,dtype=torch.float32)
                update, momentum_new = muon_update(gradient, momentum, beta=group["momentum"])
                projection_matrix_new, momentum_low_rank_new = low_rank_projection_block_iteration(momentum_new, momentum_low_rank)
                W_master = (p.float() + projection_matrix @ state["weight_residual"]) * (1 - group["lr"] * group["weight_decay"])
                W_master.add_(update.reshape(W_master.shape), alpha=-group["lr"])
                state["projection_matrix"] = projection_matrix_new
                state["momentum_buffer_low_rank"] = momentum_low_rank_new
                p.copy_(W_master.to(p.dtype))
                weight_residual_new = W_master - p.float()
                state["weight_residual"].copy_(projection_matrix_new.T @ weight_residual_new)
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
