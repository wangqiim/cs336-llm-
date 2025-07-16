import math
import torch
from torch.optim.optimizer import Optimizer

# model.parameters(),
# lr=1e-3,
# weight_decay=0.01,
# betas=(0.9, 0.999),
# eps=1e-8,
class AdamW(Optimizer):
    def __init__(
        self,
        params: torch.Tensor,
        lr: float,
        weight_decay: float,
        betas: tuple[float],
        eps: float
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
        )
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                # update
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * weight_decay * p.data
                
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        