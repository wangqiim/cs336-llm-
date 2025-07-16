import math
import torch
import torch.nn as nn
import einops

class Linear(nn.Module):
  def __init__(self, in_features: int, out_features: int, device: torch.device | None, dtype: torch.dtype | None):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    self.w = nn.Parameter(
      torch.randn((out_features, in_features), device=device, dtype=dtype)
    )
    
    self.init_params(None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = einops.einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")
    return out

  def init_params(self, weights: torch.Tensor | None):
    if weights is None:
      std = math.sqrt(2/(self.in_features + self.out_features))
      torch.nn.init.trunc_normal_(self.w.data, mean=0.0, std=std**2, a=-3*std, b=3*std)
      return
    self.w.data = nn.Parameter(weights)