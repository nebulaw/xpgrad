import numpy as np
from typing import Callable, List, Optional
from ..tensor import Tensor

class Module:
    def __call__(self, *x): return self.forward(*x)
    def forward(self): raise NotImplementedError("orward is not implemented for {}".format(type(self)))
    def parameters(self): raise NotImplementedError("{} does not have parameters".format(type(self)))

class Linear(Module):
    def __init__(self, features_in, features_out, bias=True):
        self.weights = Tensor(np.random.rand(features_in, features_out))
        self.bias = Tensor(np.random.rand(features_out, 1)) if bias else None
    def forward(self, x):
        x = self.weights.matmul(x)
        if self.bias is not None: x = x + self.bias
        return x

class Sequential(Module):
    def __init__(self, layers:Optional[List[Module]]=None):
        self.layers = layers
    def forward(self, x):
        if self.layers:
            for layer in self.layers:
                x = layer(x)
            return x
        return x

