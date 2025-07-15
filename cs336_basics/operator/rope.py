import math
import torch
import torch.nn as nn
import einops

class RoPE(nn.Module):
  def __init__(self, d_k: int, theta: float | None, max_seq_len: int, device: torch.device | None, dtype: torch.dtype | None):
    super().__init__()
    assert d_k % 2 == 0, "d_k must be divisible by 2"
    self.theta = theta if theta is not None else 1000
    self.d_k = d_k
    self.max_seq_len = max_seq_len
    half_dims = d_k // 2
    freq = theta ** (-torch.arange(0, half_dims, dtype=torch.float32) / half_dims)
    pos = torch.arange(0, max_seq_len)
    pos_freq = einops.einsum(pos, freq, "seq_len, half_dims -> seq_len half_dims")
    cos_pos_freq = torch.cos(pos_freq)
    sin_pos_freq = torch.sin(pos_freq)
    self.register_buffer("cos_pos_freq", cos_pos_freq, persistent=False)
    self.register_buffer("sin_pos_freq", sin_pos_freq, persistent=False)


  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    seq_len, feature_dim = x.size(-2), x.size(-1)
    assert seq_len <= self.max_seq_len
    assert feature_dim % 2 == 0 and feature_dim == self.d_k
    # print("===============")
    # print(x.shape)
    x = einops.rearrange(x, "... seq_len (d_k n) -> ... seq_len d_k n", n = 2)
    x_1 = x[..., 0].clone()
    x_2 = x[..., 1].clone()
    # print(f"token_positions = f{token_positions}")
    cos_base = self.cos_pos_freq[token_positions][..., :feature_dim//2]
    sin_base = self.sin_pos_freq[token_positions][..., :feature_dim//2]
    # print(f"x_1.shape = {x_1.shape}")
    # print(f"x_2.shape = {x_2.shape}")
    # print(f"cos_base.shape = {cos_base.shape}")
    # print(f"sin_base.shape = {sin_base.shape}")
    x[..., 0] = x_1 * cos_base - x_2 * sin_base
    x[..., 1] = x_1 * sin_base + x_2 * cos_base
    out = einops.rearrange(x, "... seq_len d_k n -> ... seq_len (d_k n)")
    return out
