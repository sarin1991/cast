import torch
import math
from .utils import LayerwiseOptimizerMixin


def fp32_trunc_to_bf16_grid(x_fp32: torch.Tensor) -> torch.Tensor:
    """
    Zero the lower 16 bits of the FP32 mantissa, producing an FP32 value that is
    exactly representable on the BF16 grid.
    """
    xi = x_fp32.view(torch.int32)
    xi = xi & -65536  # 0xFFFF0000
    return xi.view(torch.float32)


def decompress_bf16_to_fp32(W_bf16: torch.Tensor, P: torch.Tensor, W_lr: torch.Tensor) -> torch.Tensor:
    """
    Decompress low-rank tensor stored as:
        W_eff = W_bf16 + P @ W_lr
    where:
        W_bf16: (m, n) bf16 base weights (used in fwd/bwd storage)
        P:      (m, r) with (approximately) orthonormal columns
        W_lr:   (r, n) fp32 low-rank correction coefficients
    Returns:
        W_eff:  (m, n) fp32 effective weight
    """
    out = W_bf16.float()
    out.addmm_(P.float(), W_lr)
    return out


@torch.no_grad()
def compress_bf16_with_lr_lowbits_then_sr_(
    W_fp32: torch.Tensor,     # (m, n) fp32 ideal weight
    W_bf16: torch.Tensor,     # (m, n) bf16 tensor to write (in-place)
    P: torch.Tensor,          # (m, r) basis (~orthonormal)
    W_lr: torch.Tensor,       # (r, n) fp32 coeffs to write (in-place)
):
    """
    Compress W_fp32 into:
      - W_bf16: BF16 weights (stochastically rounded)
      - W_lr:  FP32 low-rank coefficients capturing low-bits in span(P)

    Scheme:
      1) W_hi = trunc_fp32_to_bf16_grid(W_fp32)   (exactly representable in bf16)
      2) W_lo = W_fp32 - W_hi
      3) W_lr = P^T W_lo
      4) target = W_fp32 - P W_lr   (= W_hi + (W_lo - P W_lr))
      5) W_bf16 = SR(target)
    """
    P32 = P.float()

    W_hi = fp32_trunc_to_bf16_grid(W_fp32)
    W_lo_neg = W_hi.sub_(W_fp32)

    W_lr.addmm_(P32.T, W_lo_neg, beta=0.0, alpha=-1.0) # (r, n)

    target = W_fp32
    target.addmm_(P32, W_lr, beta=1.0, alpha=-1.0)               # (m, n)

    copy_stochastic_(W_bf16, target)             # SR fp32 -> bf16 (in-place)


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
    # Polar Express coeffs computed for num_iters=5, safety_factor=2e-2, cushion=2 (offline)
    polar_express_coeffs = [
        (8.156554524902461,  -22.48329292557795,  15.878769915207462),
        (4.042929935166739,   -2.808917465908714,  0.5000178451051316),
        (3.8916678022926607,  -2.772484153217685,  0.5060648178503393),
        (3.285753657755655,   -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377,  -1.7097828382687081, 0.42323551169305323),
    ]

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Match the reference normalization (no runtime "cushion" factor)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1.0 + 2e-2) + 1e-6)

    for (a, b, c) in polar_express_coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= 0.2 * math.sqrt(max(update.size(-2), update.size(-1)))
    return update


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


class LowRankPEWRESMuonOptimizer(torch.optim.Optimizer):
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
                    if not isfloat32:
                        state["low_rank_weight_residual"] = torch.zeros_like(momentum_buffer_low_rank_init,dtype=torch.float32)
                else:
                    momentum = decompress(state["projection_matrix"], state["momentum_buffer_low_rank"])
                # Update momentum
                update = muon_update(gradient, momentum, beta=group["momentum"])
                # build weight
                if isfloat32:
                    W = p
                else:
                    W = decompress_bf16_to_fp32(p,state["projection_matrix"],state["low_rank_weight_residual"])
                W.mul_(1 - group["lr"] * group["weight_decay"])
                W.add_(update.reshape(p.shape), alpha=-group["lr"])
                # update projection matrix and momentum low rank
                q = zeropower_via_newtonschulz5(state["momentum_buffer_low_rank"].T, steps=5)
                state["projection_matrix"].copy_(zeropower_via_newtonschulz5(momentum @ q, steps=5))
                torch.matmul(state["projection_matrix"].T, momentum, out=state["momentum_buffer_low_rank"])
                if not isfloat32:
                    compress_bf16_with_lr_lowbits_then_sr_(W, p, state["projection_matrix"], state["low_rank_weight_residual"])                
        return loss

class LayerwiseLowRankPEWRESMuonOptimizer(LayerwiseOptimizerMixin,LowRankPEWRESMuonOptimizer):
    pass