from torch.autograd.graph import register_multi_grad_hook

def attach_multi_param_opt_hook(params, optimizer):
    """
    Attach a multi-parameter hook that calls optimizer.step_from_grads(params, grads)
    as soon as *all* grads for `params` are available in a backward pass.
    """
    def hook_fn(grads):
        optimizer.step_from_grads(params, grads)
        return grads
    handle = register_multi_grad_hook(params, hook_fn, mode="all")
    return handle

def attach_clear_grad_after_accumulate(param):
    """
    Clear param.grad right after autograd has finished accumulating it.
    Requires a PyTorch version with register_post_accumulate_grad_hook.
    """
    def clear_hook(p):
        p.grad = None
    return param.register_post_accumulate_grad_hook(clear_hook)
