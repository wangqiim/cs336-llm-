
import math
import torch
import torch.nn as nn
import einops

def softmax(in_features: int, dim: int, temp: float = 1.0):
  """
  带温度参数的softmax函数
  
  Args:
      in_features: 输入张量
      dim: 计算softmax的维度
      temp: 温度参数(默认1.0)，值越大分布越平滑，越小则越尖锐
  
  Returns:
      归一化的概率分布张量
  """
  in_features = in_features - in_features.max(dim = dim, keepdim=True).values
  e = torch.exp(in_features / temp)
  out = e / e.sum(dim = dim, keepdim=True)
  return out
