import os
from collections.abc import Iterable
from typing import IO, BinaryIO
import torch
import numpy as np
import numpy.typing as npt
from jaxtyping import Float, Int
from .softmax import softmax

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    epsilon = 1e-6
    p_with_grad = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in p_with_grad))

    clip_coef = max_l2_norm / (total_norm + epsilon)
    
    if total_norm > max_l2_norm:
        for p in p_with_grad:
            p.grad.data *= max_l2_norm / (total_norm + epsilon) 

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = np.array([dataset[start : start+context_length] for start in sample_starts])
    targets = np.array([dataset[start + 1 : start+context_length + 1] for start in sample_starts])
    inputs_tensor = torch.tensor(inputs, dtype=torch.int).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.int).to(device)
    return inputs_tensor, targets_tensor

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
    ):
    checkpoint = dict(
        model_param = model.state_dict(),
        optimizer_param = optimizer.state_dict(),
        iteration = iteration
    )
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None
    ):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_param"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_param"])
    return checkpoint["iteration"]

class DataLoader:
  def __init__(self, np_data_path: str, len: int, batch_size: int, context_length: int, device: str):
    self.dataset = np.memmap(np_data_path, dtype=int, mode='r', shape=(len,))
    self.batch_size = batch_size
    self.context_len = context_length
    self.device = torch.device(device)
      
  def get_batch(self):
    return get_batch(self.dataset, self.batch_size, self.context_len, self.device)

def sample(logsitcs: Float[torch.Tensor, "vocab_size"], temp: float, top_p: float):
    assert top_p >= 0.0 and top_p <= 1.0
    probs = softmax(logsitcs, dim=-1, temp=temp)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs <= top_p
    mask = torch.cat([torch.ones(1, dtype=torch.bool, device=mask.device), mask[1:]])  # 确保至少选一个词
    
    selected_probs = sorted_probs[mask]
    selected_indices = sorted_indices[mask]
    selected_probs /= selected_probs.sum()
    next_token = torch.multinomial(selected_probs, 1)
    return selected_indices[next_token]
