from collections.abc import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    epsilon = 1e-6
    p_with_grad = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in p_with_grad))

    clip_coef = max_l2_norm / (total_norm + epsilon)
    
    if total_norm > max_l2_norm:
        for p in p_with_grad:
            p.grad.data *= max_l2_norm / (total_norm + epsilon) 
    
    