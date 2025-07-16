
import math
import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Int
from .softmax import softmax
from .rope import RoPE
from .linear import Linear
from .rmsnorm import RMSNorm
from .swiglu import Swiglu
from .embedding import Embedding

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
  ) -> torch.Tensor:
  assert Q.size(-1) == K.size(-1), f"Q{Q.shape}, V{K.shape}, features dim must be equal"
  assert K.size(-2) == V.size(-2), f"K{K.shape}, V{V.shape}, seq_len must be equal"
  d_k = Q.size(-1)
  score = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
  if mask is not None:
    score.masked_fill_(mask == False, float("-inf"))
  qk_probs = softmax(score / math.sqrt(d_k), dim=-1)
  out = einops.einsum(qk_probs, V, "... queries keys, ... keys d_v -> ... queries d_v") 
  return out

class MultiHeadAttention(nn.Module):
  def __init__(self, 
    d_model: int,
    num_heads: int,
    theta: float | None = None,
    max_seq_len: int | None = None,
    device: torch.device | None = None, 
    dtype: torch.dtype | None = None
  ):
    super().__init__()
    assert d_model % num_heads == 0, "d_model must be dived by num_heads"
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = self.d_model // self.num_heads
    self.q_proj = Linear(d_model, d_model, device, dtype)
    self.k_proj = Linear(d_model, d_model, device, dtype)
    self.v_proj = Linear(d_model, d_model, device, dtype)
    self.o_proj = Linear(d_model, d_model, device, dtype)
    if theta is not None and max_seq_len is not None:
      self.rope = RoPE(self.head_dim, theta, max_seq_len, device, dtype)
    else:
      self.rope = None

  def forward(
      self, 
      in_features: Float[torch.Tensor, "... sequence_length d_in"],
      token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
    B = in_features.size(0)
    seq_len = in_features.size(-2)
    Q = einops.rearrange(self.q_proj(in_features), "... seq_len (num_head d) -> ... num_head seq_len d", num_head = self.num_heads)
    K = einops.rearrange(self.k_proj(in_features), "... seq_len (num_head d) -> ... num_head seq_len d", num_head = self.num_heads)
    V = einops.rearrange(self.v_proj(in_features), "... seq_len (num_head d) -> ... num_head seq_len d", num_head = self.num_heads)
    
    if self.rope:
      if token_positions is None:
        token_positions = torch.arange(0, seq_len)
        token_positions = einops.repeat(token_positions, "seq_len -> b seq_len", b = B)
      Q = self.rope(Q, token_positions)
      K = self.rope(K, token_positions)
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(Q.device)
    out = scaled_dot_product_attention(Q, K, V, mask)
    out = einops.rearrange(out, "... num_head seq_len d -> ... seq_len (num_head d)")
    return self.o_proj(out)

  def init_params(
    self,
    q_proj_weight: Float[torch.Tensor, " d_k d_in"],
    k_proj_weight: Float[torch.Tensor, " d_k d_in"],
    v_proj_weight: Float[torch.Tensor, " d_v d_in"],
    o_proj_weight: Float[torch.Tensor, " d_model d_v"]    
  ):
    self.q_proj.init_params(q_proj_weight)
    self.k_proj.init_params(k_proj_weight)
    self.v_proj.init_params(v_proj_weight)
    self.o_proj.init_params(o_proj_weight)

class TransformerBlock(nn.Module):
  def __init__(self, 
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    device: torch.device | None = None, 
    dtype: torch.dtype | None = None
  ):
    super().__init__()
    self.ln1 = RMSNorm(d_model)
    self.mha = MultiHeadAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
    
    self.ln2 = RMSNorm(d_model)
    self.ffn = Swiglu(d_model, d_ff, device, dtype)

  def forward(self, in_features: Float[torch.Tensor, " batch sequence_length d_model"]) -> torch.Tensor:
    mid = in_features + self.mha(self.ln1(in_features))
    output = mid + self.ffn(self.ln2(mid))
    return output

  def init_params(self, weights: dict[str, torch.Tensor]):
    self.ln1.init_params(weights["ln1.weight"])
    self.ln2.init_params(weights["ln2.weight"])
    self.mha.init_params(
      weights["attn.q_proj.weight"], 
      weights["attn.k_proj.weight"], 
      weights["attn.v_proj.weight"], 
      weights["attn.output_proj.weight"])
    self.ffn.init_params(weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"])

class TransformerLM(nn.Module):
  def __init__(self,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: torch.device | None = None, 
    dtype: torch.dtype | None = None
  ):
    super().__init__()
    self.embedding = Embedding(vocab_size, d_model, device, dtype)
    self.layers = nn.ModuleList(
      TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
      for _ in range(num_layers)
    )
    self.ln_final = RMSNorm(d_model)
    self.lm_head = Linear(d_model, vocab_size, device, dtype)
    self.num_layers = num_layers

  def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]) -> torch.Tensor:
    x = self.embedding(in_indices)
    for block in self.layers:
      x = block(x)
    return self.lm_head(self.ln_final(x))

  def init_params(self, weights: dict[str, torch.Tensor]):
    self.embedding.init_params(weights["token_embeddings.weight"])
    self.ln_final.init_params(weights["ln_final.weight"])
    self.lm_head.init_params(weights["lm_head.weight"])
    for i in range(self.num_layers):
      w = {}
      w["ln1.weight"] = weights[f"layers.{i}.ln1.weight"]
      w["ln2.weight"] = weights[f"layers.{i}.ln2.weight"]
      w["attn.q_proj.weight"] = weights[f"layers.{i}.attn.q_proj.weight"]
      w["attn.k_proj.weight"] = weights[f"layers.{i}.attn.k_proj.weight"]
      w["attn.v_proj.weight"] = weights[f"layers.{i}.attn.v_proj.weight"]
      w["attn.output_proj.weight"] = weights[f"layers.{i}.attn.output_proj.weight"]
      w["ffn.w1.weight"] = weights[f"layers.{i}.ffn.w1.weight"]
      w["ffn.w2.weight"] = weights[f"layers.{i}.ffn.w2.weight"]
      w["ffn.w3.weight"] = weights[f"layers.{i}.ffn.w3.weight"]
      self.layers[i].init_params(w)
