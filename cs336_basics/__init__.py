import importlib.metadata
from .BPETokenizer import train_bpe
from .BPETokenizer import BPETokenizer
from .operator import Embedding, Linear, RMSNorm, Swiglu, RoPE, MultiHeadAttention, TransformerBlock, TransformerLM
from .operator import softmax, scaled_dot_product_attention, silu

__version__ = importlib.metadata.version("cs336_basics")
