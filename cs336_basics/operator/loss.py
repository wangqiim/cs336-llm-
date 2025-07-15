import torch
from jaxtyping import Float, Int
import einops

def cross_entropy(
  inputs: Float[torch.Tensor, "batch_size vocab_size"],
  targets: Int[torch.Tensor, "batch_size"]
) -> Float[torch.Tensor, ""]:
  stable_inputs = inputs - einops.reduce(inputs, "batch_size vocab_size -> batch_size 1", "max")
  log_probs = stable_inputs - torch.log(einops.reduce(torch.exp(stable_inputs), "batch_size vocab_size -> batch_size 1", "sum"))
  return -log_probs[torch.arange(inputs.size(0)), targets].mean()
