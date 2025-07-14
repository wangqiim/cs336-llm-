import importlib.metadata
from .BPETokenizer import train_bpe
from .BPETokenizer import BPETokenizer
from .operator import Embedding, Linear, RMSNorm, Swiglu, RoPE

__version__ = importlib.metadata.version("cs336_basics")
