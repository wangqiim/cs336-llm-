import math
import torch
import torch.nn as nn
import einops

class Embedding(nn.Module):
  def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None, dtype: torch.dtype | None):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim

    self.w = nn.Parameter(
      torch.randn((num_embeddings, embedding_dim), device=device, dtype=dtype)
    )
    self.init_params(None)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    out = self.w[token_ids]
    return out

  def init_params(self, weights: torch.Tensor | None):
    if weights is None:
      torch.nn.init.trunc_normal_(self.w.data, mean=0.0, std=1, a=-3, b=3)
      return
    self.w.data = nn.Parameter(weights)