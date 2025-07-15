
import math
import torch
import torch.nn as nn
import einops

def softmax(in_features: int, dim: int):
  in_features = in_features - in_features.max(dim = dim, keepdim=True).values
  e = torch.exp(in_features)
  out = e / e.sum(dim = dim, keepdim=True)
  return out
