import math
import torch
import torch.nn as nn
import einops

class RMSNorm(nn.Module):
  def __init__(self,  d_model: int, eps: float = 1e-5, device: torch.device | None = None):
    super().__init__()
    self.d_model = d_model
    self.eps = eps

    self.g = nn.Parameter(
      torch.randn((self.d_model), device=device, dtype=torch.float32)
    )
    self.init_params(None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result_type = x.dtype
    x = x.to(torch.float32)
    rms = torch.sqrt(einops.reduce(x * x, '... d -> ... 1', 'sum') / self.d_model + self.eps)
    out = x / rms * self.g

    return out.to(result_type)

  def init_params(self, weights: torch.Tensor | None):
    if weights is None:
      self.g.data.fill_(1.0)
      return
    self.g.data = nn.Parameter(weights)