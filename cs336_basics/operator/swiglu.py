
import math
import torch
import torch.nn as nn
from .linear import Linear
import einops

def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class Swiglu(nn.Module):
  def __init__(self,  d_model: int, d_ff: int, device: torch.device | None, dtype: torch.dtype | None):
    super().__init__()
    self.d_model = d_model
    self.d_ff = d_ff

    self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
    self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.linear2(silu(self.linear1(x)) * self.linear3(x))
    return out

  def init_params(self, w1: torch.Tensor | None, w2: torch.Tensor | None, w3: torch.Tensor | None):
    self.linear1.init_params(w1)
    self.linear2.init_params(w2)
    self.linear3.init_params(w3)
