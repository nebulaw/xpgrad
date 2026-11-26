from __future__ import annotations
import numpy as np
from typing import List, Union, Optional, Tuple

# TODO:

# this level of abstraction is nearly killing the machine.
# like this a lot? :3, i mean its really runtime dependent
# code and I don't think its clean and easy to understand
# but the beauty is inside of it. I see this way kinda

class Function:
    def __init__(self):
        self.parents: Tuple[Tensor, ...] = ()
    @classmethod
    def apply(cls, *x:"Tensor") -> "Tensor":
        fn = cls()
        out = Tensor(fn.forward(*[t.data for t in x]))
        out.requires_grad = any(t.requires_grad for t in x)
        if out.requires_grad: fn.parents, out.grad_fn = x, fn
        return out
    def __repr__(self): return "<{}>".format(type(self))
    def forward(self, *xs: np.ndarray) -> np.ndarray: raise NotImplementedError("Forward is not implemented for {}".format(type(self).__name__))
    def backward(self, g: np.ndarray) -> Tuple[Optional[np.ndarray], ...]: raise NotImplementedError("Backward is not implemented for {}".format(type(self).__name__))

def _tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)

class Tensor:
    __slots__ = ("data", "grad", "grad_fn", "requires_grad")
    def __init__(self, data:Union[np.ndarray, int, float], requires_grad:Optional[bool]=None):
        if isinstance(data, (int, float)): data = np.array([data])
        elif isinstance(data, list): data = np.array(data)
        elif isinstance(data, np.ndarray): data = data
        else: assert False, "data must be of type: np.ndarray, int, float, list"
        self.data = data
        self.grad:Optional['Tensor'] = None
        self.grad_fn:Optional[Function] = None
        self.requires_grad = requires_grad
    @property
    def dtype(self): return self.data.dtype
    @property
    def shape(self): return self.data.shape
    @property
    def T(self): return self.assign(self.data.T)
    # binary ops
    def add(self, x) -> "Tensor": return Add.apply(self, _tensor(x))
    def mul(self, x) -> "Tensor": return Mul.apply(self, _tensor(x))
    def matmul(self, x) -> "Tensor": return MatMul.apply(self, _tensor(x))
    # operators, composed from above so they wont' require backward of their own
    def __add__(self, x) -> "Tensor": return self.add(x)
    def __radd(self, x) -> "Tensor": return _tensor(x).add(self)
    def __mul__(self, x) -> "Tensor": return self.mul(x)
    def __rmul__(self, x) -> "Tensor": return _tensor(x).mul(self)
    def __matmul__(self, x) -> "Tensor": return self.matmul(_tensor(x))
    def __neg__(self) -> "Tensor": return self.mul(-1.0)
    def __sub__(self, x) -> "Tensor": return self.add(-_tensor(x))
    def __rsub__(self, x) -> "Tensor": return _tensor(x).add(-self)
    def __truediv__(self, x) -> "Tensor": return self.mul(_tensor(x).reciprocal())
    def __rtruediv__(self, x) -> "Tensor": return _tensor(x).mul(self.reciprocal())
    # Unary ops
    def sum(self) -> "Tensor": return Sum.apply(self)
    def reciprocal(self) -> "Tensor": return Reciprocal.apply(self)
    def relu(self) -> "Tensor": return ReLU.apply(self)
    def tanh(self) -> "Tensor": return Tanh.apply(self)
    def sigmoid(self) -> "Tensor": return Sigmoid.apply(self)
    def exp(self) -> "Tensor": return Exp.apply(self)
    def topowalk(self) -> List['Tensor']:
        def _topowalk(node, visited):
            visited.add(node)
            if getattr(node, "grad_fn", None):
                for p in node.grad_fn.parents:
                    if p not in visited:
                        yield from _topowalk(p, visited)
                yield node
        return list(_topowalk(self, set()))
    def backward(self) -> None:
        if self.grad_fn is None:
            return
        if self.grad is None:
            assert self.data.size == 1, "grad can only be implicitly created for scalar tensors"
            self.grad = Tensor(1.0, requires_grad=False)

        assert self.grad is not None

        for tt in reversed(self.topowalk()):
            if tt.grad is None: raise ValueError("{} grad is None".format(tt))
            assert tt.grad_fn is not None, "{} grad_fn is None".format(tt)
            grads = tt.grad_fn.backward(tt.grad.data)
            grads = [Tensor(g, requires_grad=False) if g is not None else None for g in ([grads] if not isinstance(grads, (List, Tuple)) else grads)]
            for parent, grad in zip(tt.grad_fn.parents, grads):
                if grad is not None and parent.requires_grad:
                    assert grad.shape == parent.shape, "grad shape must match tensor: %r != %r" % (grad.shape, parent.data.shape, )
                    parent.grad = grad if parent.grad is None else parent.grad + grad
            # commented out temporarily causes __repr__ error on backward()
            # del tt.grad_fn
            tt.grad_fn = None
    def __repr__(self): return "Tensor(data={}, grad={}, grad_fn={})".format(self.data, self.grad, self.grad_fn)
    def assign(self, x): del self.data; self.data = x
    def reshape(self, *shape): return self.assign(self.data.reshape(*shape))
    def view(self, *shape): return self.reshape(*shape)
    def tolist(self): return self.data.tolist()


class Add(Function):
    def forward(self, x, y): return x + y
    def backward(self, g): return g, g

class Mul(Function):
    def forward(self, x, y): self.x, self.y = x, y; return x * y
    def backward(self, g): return g * self.y, g * self.x

class Reciprocal(Function):
    def forward(self, x): self.out = 1.0 / x; return self.out
    def backward(self, g): return (-g * self.out * self.out,)

class MatMul(Function):
    def forward(self, x, y): self.x, self.y = x, y; return x @ y
    def backward(self, g): return g @ self.y.T, self.x.T @ g

class Sum(Function):
    def forward(self, x): self.shape = x.shape; return np.array([x.sum()])
    def backward(self, g): return (np.broadcast_to(g, self.shape).copy(),)

class ReLU(Function):
    def forward(self, x): self.mask = x > 0; return np.maximum(x, 0)
    def backward(self, g): return (g * self.mask,)

class Tanh(Function):
    def forward(self, x): self.out = np.tanh(x); return self.out
    def backward(self, g): return (g * (1 - self.out * self.out),)

class Sigmoid(Function):
    def forward(self, x): self.out = 0.5 * (1 + np.tanh(0.5 * x)); return self.out # this one wins over 1 / (1 + exp(-x)) cz never overflows
    def backward(self, g): return (g * self.out * (1 - self.out),)

class Exp(Function):
    def forward(self, x): self.out = np.exp(x); return self.out
    def backward(self, g): return (g * self.out,)

