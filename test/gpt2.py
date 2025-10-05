from typing import Optional, Union
import argparse
from tqdm import trange
import numpy as np
import tiktoken
from xpgrad import Tensor
import xpgrad as xp
import xpgrad.nn as nn
import xpgrad.nn.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        self.attn = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

    def __call__(self, x:Tensor, start_pos, mask):
        if mask is not None or start_pos.val == 0:
            start_pos = start_pos.val

    

