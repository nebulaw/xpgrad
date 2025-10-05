from __future__ import annotations
import numpy as np
from typing import List, Union, Optional

# TODO:
# (1) add more functions

# this level of abstraction is nearly killing the machine.
# like this a lot? :3, i mean its really runtime dependent
# code and I don't think its clean and easy to understand
# but the beauty is inside of it. I see this way kinda

class Function:
    def __init__(self, *x:'Tensor'):
        super().__init__()
        self.required_grads = tuple(t.requires_grad for t in x)
        self.requires_grad = True if any(t.requires_grad for t in x) else (None if None in self.required_grads else False)
        if self.requires_grad: self.parents = x
    @classmethod
    def apply(cls, *x:'Tensor'):
        fnc = cls(*x)
        t = Tensor.__new__(Tensor)
        t.data = fnc.forward(*[tens.data for tens in x])
        t.grad = None
        t.grad_fn = fnc if fnc.requires_grad else None
        t.requires_grad = fnc.requires_grad
        return t
    def __repr__(self): return "<{}>".format(type(self))
    def forward(self, *_) -> "Tensor": raise NotImplementedError("Forward is not implemented for {}".format(type(self)))
    def backward(self, *_) -> "Tensor": raise NotImplementedError("Backward is not implemented for {}".format(type(self)))

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
    def __add__(self, x) -> "Tensor": return self.add(x)
    def __mul__(self, x) -> "Tensor": return self.dot(x)
    def __matmul__(self, x) -> "Tensor": return self.matmul(x)
    # binary ops
    def add(self, x) -> "Tensor": return Add.apply(self, x)
    def mul(self, x) -> "Tensor": return Mul.apply(self, x)
    def dot(self, x) -> "Tensor": return Dot.apply(self, x)
    def matmul(self, x) -> "Tensor": return MatMul.apply(self, x)
    # Unary ops
    def sum(self) -> "Tensor": return Sum.apply(self)
    def relu(self) -> "Tensor": return ReLU.apply(self)
    def tanh(self) -> "Tensor": return Tanh.apply(self)
    def sig(self) -> "Tensor": return Sig.apply(self)
    def topowalk(self) -> List['Tensor']:
        def _topowalk(node, visited):
            visited.add(node)
            if getattr(node, "grad_fn", None):
                for p in node.grad_fn.parents:
                    if p not in visited:
                        yield from _topowalk(p, visited)
                yield node
        return list(_topowalk(self, set()))
    def backward(self):
        if self.grad_fn is None:
            return
        if self.grad is None:
            assert self.data.size == 1, "grad can only be implicitly created for scalar tensors"
            self.grad = Tensor(1.0, requires_grad=False)

        assert self.grad is not None

        for tt in reversed(self.topowalk()):
            if tt.grad is None: raise ValueError("{} grad is None".format(tt))
            assert tt.grad_fn is not None, "{} grad_fn is None".format(tt)
            grads = tt.grad_fn.backward(tt.grad)
            grads = [Tensor(g, requires_grad=False) if g is not None else None for g in ([grads] if not isinstance(grads, (list, tuple)) else grads)]
            for parent, grad in zip(tt.grad_fn.parents, grads):
                if grad is not None and parent.requires_grad:
                    assert grad.shape == parent.shape, "grad shape must match tensor: %r != %r" % (grad.shape, parent.data.shape, )
                    parent.grad = grad if parent.grad is None else parent.grad + grad
            del tt.grad_fn
        return self
    def __repr__(self): return "Tensor(data={}, grad={}, grad_fn={})".format(self.data, self.grad, self.grad_fn)
    def assign(self, x): del self.data; self.data = x
    def reshape(self, *shape): return self.assign(self.data.reshape(*shape))
    def view(self, *shape): return self.reshape(*shape)
    def tolist(self): return self.data.tolist()

class Mul(Function):
    def forward(self, x, y):
        self.x, self.y = Tensor(x), Tensor(y)
        return x * y
    def backward(self, grad):
        return self.y.data * grad.data if self.required_grads[0] else None, \
                self.x.data * grad.data if self.required_grads[1] else None

class Add(Function):
    def forward(self, x, y):
        self.x, self.y = Tensor(x), Tensor(y)
        return x + y
    def backward(self, grad):
        return grad.data if self.required_grads[0] else None, \
                grad.data if self.required_grads[1] else None

class Dot(Function):
    def forward(self, x, y):
        self.x, self.y = Tensor(x), Tensor(y)
        return x.dot(y)
    def backward(self, grad):
        return grad.data.dot(self.y.data.T) if self.required_grads[0] else None, \
                grad.data.T.dot(self.x.data).T if self.required_grads[1] else None

class MatMul(Function):
    def forward(self, x, y):
        self.x, self.y = Tensor(x), Tensor(y)
        return np.matmul(x, y)
    def backward(self, grad):
        return np.matmul(grad.data, self.y.data.T) if self.required_grads[0] else None, \
                np.matmul(self.x.data.T, grad.data) if self.required_grads[1] else None

class Sum(Function):
    def forward(self, x) -> np.ndarray:
        self.shape = x.shape
        return np.array([x.sum()])
    def backward(self, grad):
        return np.zeros(self.shape) + grad.data if self.required_grads[0] else None

class ReLU(Function):
    def forward(self, x):
        self.x = Tensor(x)
        return np.maximum(x, 0)
    def backward(self, grad):
        gx = grad.data.copy()
        gx[self.x.data < 0] = 0
        return gx if self.required_grads[0] else None

class Tanh(Function):
    def forward(self, x):
        self.tx = Tensor(np.tanh(x))
        return self.tx
    def backward(self, grad):
        return (1 - self.tx.data**2) * grad.data if self.required_grads[0] else None

class Sig(Function):
    def forward(self, x: np.ndarray):
        self.s = Tensor(1 / (1 + np.exp(-x)))
        return self.s
    def backward(self, grad):
        return (self.s.data / (1 - self.s.data)) * grad.data if self.required_grads[0] else None


