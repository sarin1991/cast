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
def low_rank_projection_block_iteration(P: torch.Tensor, M: torch.Tensor, M_lr: torch.Tensor, q_buf: torch.Tensor, r_buf: torch.Tensor):
    """
    One warm-started block iteration that updates the low rank projection matrix.

    Inputs:
      P:      (m, r) projection matrix
      M:      (m, n) fp32
      M_lr: (r, n) low rank momentum
      q_buf: (n,r) buffer
      r_buf: (r,r) buffer

    Returns:
      None # inplace updates P & M_lr
    """
    torch.linalg.qr(M_lr.T, mode="reduced",out=(q_buf,r_buf)) # q = (n,r)
    torch.linalg.qr(M @ q_buf, mode="reduced",out=(P,r_buf)) # q = (m,r)
    # Store new low rank momentum
    torch.matmul(P.T, M, out=M_lr)  # (r, n)


class AllLowRankMuonOptimizer(torch.optim.Optimizer):
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
                gradient = p.grad.detach()
                # Shapes
                m, n = gradient.shape
                r = group["momentum_rank"]
                if len(state) == 0:
                    projection_matrix_init = torch.randn((gradient.shape[0], r), device=p.device, dtype=torch.float32)
                    state["projection_matrix"] = _orthonormalize_columns(projection_matrix_init)
                    momentum_buffer_low_rank_init = torch.randn((r, gradient.shape[1]), device=p.device, dtype=torch.float32)
                    state["momentum_buffer_low_rank"] = momentum_buffer_low_rank_init
                    state["weight_residual"] = torch.zeros_like(momentum_buffer_low_rank_init,dtype=torch.float32)
                    momentum = torch.zeros_like(p,dtype=torch.float32)
                else:
                    momentum = decompress(state["projection_matrix"], state["momentum_buffer_low_rank"])
                decay = (1 - group["lr"] * group["weight_decay"])
                # Buffers
                q_buf = torch.empty((n,r),device=state["projection_matrix"].device,dtype=state["projection_matrix"].dtype)
                r_buf = torch.empty((r,r),device=state["projection_matrix"].device,dtype=state["projection_matrix"].dtype)
                W = p.float()
                W.addmm_( state["projection_matrix"].float(), state["weight_residual"], beta=decay, alpha=decay)
                beta = group["momentum"]
                momentum.lerp_(gradient, 1 - beta)
                update = gradient.lerp_(momentum, beta)
                low_rank_projection_block_iteration(state["projection_matrix"], momentum, state["momentum_buffer_low_rank"],q_buf,r_buf)
                torch.linalg.qr(state["projection_matrix"].T @ update, mode="reduced",out=(q_buf,r_buf))
                torch.matmul(state["projection_matrix"], q_buf.T, out=update)
                W.add_(update.reshape(W.shape), alpha=-group["lr"])
                p.copy_(W)
                W.sub_(p)
                torch.matmul(state["projection_matrix"].T.float(), W, out=state["weight_residual"])
        return loss

class LayerwiseAllLowRankMuonOptimizer(AllLowRankMuonOptimizer):
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
