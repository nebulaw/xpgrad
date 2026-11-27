import math
from typing import Iterator

import numpy as np

from xpgrad.tensor import DEFAULT_DTYPE, Tensor


def _walk(value):
  if isinstance(value, Tensor):
    if value.requires_grad: yield value
  elif isinstance(value, Module):
    for v in vars(value).values(): yield from _walk(v)
  elif isinstance(value, (list, tuple)):
    for v in value: yield from _walk(v)

class Module:
  def __call__(self, *x): return self.forward(*x)
  def forward(self, *x): raise NotImplementedError("forward is not implemented for {}".format(type(self).__name__))
  def params(self) -> Iterator[Tensor]:
    seen = set()
    for p in _walk(self):
      if id(p) not in seen: seen.add(id(p)); yield p

class Linear(Module):
  def __init__(self, features_in, features_out, bias=True, gain=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    bound = gain * math.sqrt(6 / (features_in + features_out))
    w = rng.uniform(-bound, bound, (features_in, features_out))
    self.weight = Tensor(w.astype(DEFAULT_DTYPE), requires_grad=True)
    self.bias = Tensor(np.zeros(features_out, DEFAULT_DTYPE), requires_grad=True) if bias else None
  def forward(self, x): y = x @ self.weight; return y if self.bias is None else y + self.bias

class Sequential(Module):
  def __init__(self, *layers): self.layers = list(layers)
  def forward(self, x):
    for layer in self.layers: x = layer(x)
    return x


