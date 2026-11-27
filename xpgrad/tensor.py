from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Callable

from xpgrad import autograd

# I think the code is in base shape for now

class Function:
  parents: Tuple[Tensor, ...]
  forward: Callable[..., np.ndarray]
  backward: Callable[..., Tuple[np.ndarray, ...]]
  @classmethod
  def apply(cls, *x:"Tensor", **options) -> "Tensor":
    fn = cls(**options)
    out = Tensor(fn.forward(*[t.data for t in x]))
    # here we added the autograd flag
    if autograd.is_enabled() and any(t.requires_grad for t in x):
      fn.parents, out.grad_fn, out.requires_grad = x, fn, True
    return out
  def __repr__(self): return "<{}>".format(type(self).__name__)

# unary/binary split for typechecking, options moved to the init
class Unary(Function):
  def forward(self, x:np.ndarray) -> np.ndarray: raise NotImplementedError
  def backward(self, g:np.ndarray) -> Tuple[np.ndarray]: raise NotImplementedError

class Binary(Function):
  def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray: raise NotImplementedError
  def backward(self, g:np.ndarray) -> Tuple[np.ndarray, np.ndarray]: raise NotImplementedError

DEFAULT_DTYPE=np.float32
def _tensor(x):
  return x if isinstance(x, Tensor) else Tensor(x)

class Tensor:
  __slots__ = ("data", "grad", "grad_fn", "requires_grad")
  def __init__(self, data, requires_grad: bool = False):
    if isinstance(data, Tensor): raise TypeError("data is already a Tensor")
    floating = isinstance(data, (np.ndarray, np.generic)) and np.issubdtype(data.dtype, np.floating)
    self.data: np.ndarray = np.asarray(data) if floating else np.asarray(data, dtype=DEFAULT_DTYPE)
    self.grad:Optional['Tensor'] = None
    self.grad_fn:Optional[Function] = None
    self.requires_grad = requires_grad
  def __repr__(self): return "Tensor(data={}, grad={}, grad_fn={})".format(self.data, self.grad, self.grad_fn)
  @property
  def dtype(self): return self.data.dtype
  @property
  def shape(self): return self.data.shape
  @property
  def ndim(self): return self.data.ndim
  @property
  def T(self) -> "Tensor": return self.permute(*reversed(range(self.ndim)))
  # binary ops
  def add(self, x) -> "Tensor": return Add.apply(self, _tensor(x))
  def mul(self, x) -> "Tensor": return Mul.apply(self, _tensor(x))
  def matmul(self, x) -> "Tensor": return MatMul.apply(self, _tensor(x))
  # operators, composed from above so they wont' require backward of their own
  def __add__(self, x) -> "Tensor": return self.add(x)
  def __radd__(self, x) -> "Tensor": return _tensor(x).add(self)
  def __mul__(self, x) -> "Tensor": return self.mul(x)
  def __rmul__(self, x) -> "Tensor": return _tensor(x).mul(self)
  def __matmul__(self, x) -> "Tensor": return self.matmul(_tensor(x))
  def __neg__(self) -> "Tensor": return self.mul(-1.0)
  def __sub__(self, x) -> "Tensor": return self.add(-_tensor(x))
  def __rsub__(self, x) -> "Tensor": return _tensor(x).add(-self)
  def __truediv__(self, x) -> "Tensor": return self.mul(_tensor(x).reciprocal())
  def __rtruediv__(self, x) -> "Tensor": return _tensor(x).mul(self.reciprocal())
  # Unary ops
  def sum(self, axis=None, keepdims=False) -> "Tensor": return Sum.apply(self, axis=axis, keepdims=keepdims)
  def reciprocal(self) -> "Tensor": return Reciprocal.apply(self)
  def relu(self) -> "Tensor": return ReLU.apply(self)
  def tanh(self) -> "Tensor": return Tanh.apply(self)
  def sigmoid(self) -> "Tensor": return Sigmoid.apply(self)
  def exp(self) -> "Tensor": return Exp.apply(self)
  def reshape(self, *shape) -> "Tensor":
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    return Reshape.apply(self, shape=shape)
  def view(self, *shape) -> "Tensor": return self.reshape(*shape)
  def permute(self, *axes) -> "Tensor": return Permute.apply(self, axes=axes)
  def item(self) -> float: return self.data.item()
  def tolist(self): return self.data.tolist()
  # and here is the refactored version of backward
  def backward(self, grad=None) -> None:
    for leaf, g in autograd.backward(self, None if grad is None else _tensor(grad).data):
      leaf.grad = Tensor(g) if leaf.grad is None else Tensor(leaf.grad.data + g)

def unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
  while g.ndim > len(shape): g = g.sum(axis=0)
  for axis, n in enumerate(shape):
    if n == 1 and g.shape[axis] != 1: g = g.sum(axis=axis, keepdims=True)
  return g

class Add(Binary):
  def forward(self, x, y):
    self.x_shape, self.y_shape = x.shape, y.shape
    return x + y
  def backward(self, g):
    return unbroadcast(g, self.x_shape), unbroadcast(g, self.y_shape)

class Mul(Binary):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x * y
  def backward(self, g):
    return unbroadcast(g * self.y, self.x.shape), unbroadcast(g * self.x, self.y.shape)

class MatMul(Binary):
  def forward(self, x, y):
    if x.ndim < 2 or y.ndim < 2: raise ValueError("matmul requires inputs to have 2 or more dims")
    self.x, self.y = x, y
    return x @ y
  def backward(self, g):
    xt, yt = np.swapaxes(self.x, -1, -2), np.swapaxes(self.y, -1, -2)
    return unbroadcast(g @ yt, self.x.shape), unbroadcast(xt @ g, self.y.shape)

class Reciprocal(Unary):
  def forward(self, x): self.out = 1.0 / x; return self.out
  def backward(self, g): return (-g * self.out * self.out,)

class Sum(Unary):
  def __init__(self, axis=None, keepdims=False): self.axis, self.keepdims = axis, keepdims
  def forward(self, x):
    self.in_shape = x.shape
    return np.sum(x, axis=self.axis, keepdims=self.keepdims)
  def backward(self, g):
    if self.axis is not None and not self.keepdims: g = np.expand_dims(g, self.axis)
    return (np.broadcast_to(g, self.in_shape).copy(),)

class Reshape(Unary):
  def __init__(self, shape): self.shape = shape
  def forward(self, x): self.in_shape = x.shape; return x.reshape(self.shape)
  def backward(self, g): return (g.reshape(self.in_shape),)

class Permute(Unary):
  def __init__(self, axes): self.axes = axes
  def forward(self, x): self.inverse = np.argsort(self.axes); return np.transpose(x, self.axes)
  def backward(self, g): return (np.transpose(g, self.inverse),)

class ReLU(Unary):
  def forward(self, x): self.mask = x > 0; return np.maximum(x, 0)
  def backward(self, g): return (g * self.mask,)

class Tanh(Unary):
  def forward(self, x): self.out = np.tanh(x); return self.out
  def backward(self, g): return (g * (1 - self.out * self.out),)

class Sigmoid(Unary):
  def forward(self, x): self.out = 0.5 * (1 + np.tanh(0.5 * x)); return self.out # this one wins over 1 / (1 + exp(-x)) cz never overflows
  def backward(self, g): return (g * self.out * (1 - self.out),)

class Exp(Unary):
  def forward(self, x): self.out = np.exp(x); return self.out
  def backward(self, g): return (g * self.out,)

