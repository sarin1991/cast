from torch.autograd.graph import register_multi_grad_hook
import torch

def print_mem_stats_func(state):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved()  / 1024**3
    pa = torch.cuda.max_memory_allocated() / 1024**3
    pr = torch.cuda.max_memory_reserved()  / 1024**3
    print(f"[{state}] a/r {a:.2f}/{r:.2f}G | peak a/r {pa:.2f}/{pr:.2f}G", flush=True)

def attach_multi_param_opt_hook(params, optimizer, print_memory_stats):
    """
    Attach a multi-parameter hook that calls optimizer.step_from_grads(params, grads)
    as soon as *all* grads for `params` are available in a backward pass.
    """
    def hook_fn(grads):
        if print_memory_stats:
            torch.cuda.reset_peak_memory_stats()
            print_mem_stats_func(f"pre optimizer step")
        optimizer.step_from_grads(params, grads)
        if print_memory_stats:
            print_mem_stats_func(f"post optimizer step")
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
