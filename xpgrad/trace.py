from math import prod
from xpgrad.graph import Node, node, Op, const


# tracer records every op it has as a graph
class Tracer:
  __array_ufunc__ = None # hand operators back to us, we own them!
  def __init__(self, n: Node): self.node = n
  @property
  def shape(self): return self.node.shape
  @property
  def dtype(self): return self.node.dtype
  @property
  def ndim(self): return len(self.node.shape)
  # primitives
  def __repr__(self): return "Tracer({})".format(self.node)
  def __add__(self, x): return _binary(Op.ADD, self, x)
  def __radd__(self, x): return _binary(Op.ADD, x, self)
  def __mul__(self, x): return _binary(Op.MUL, self, x)
  def __rmul__(self, x): return _binary(Op.MUL, x, self)
  def reshape(self, *shape): return _unary_dim(Op.RESHAPE, self, shape)
  def expand(self, *shape): return _unary_dim(Op.EXPAND, self, shape)
  def broadcast_to(self, shape):
    x = self
    if x.ndim < len(shape): x = x.reshape((1,) * (len(shape) - x.ndim) + x.shape)
    if x.shape != shape: x = x.expand(shape)
    return x
  # element-wise prims
  def exp(self): return Tracer(node(Op.EXP, self.node))
  def log(self): return Tracer(node(Op.LOG, self.node))
  def sqrt(self): return Tracer(node(Op.SQRT, self.node))
  def tanh(self): return Tracer(node(Op.TANH, self.node))
  def reciprocal(self): return Tracer(node(Op.RECIP, self.node))
  def cast(self, dtype): return Tracer(node(Op.CAST, self.node, attr=dtype))
  def maximum(self, x): return _binary(Op.MAX, self, x)
  # composed ones
  def __neg__(self): return self * -1.0
  def __sub__(self, x): return self + -_lift(x, self.dtype)
  def __rsub__(self, x): return -self + x # '+' will lift x if necessary
  def __truediv__(self, x): return self * _lift(x, self.dtype).reciprocal()
  def __rtruediv__(self, x): return self.reciprocal() * x # as well '*'
  def relu(self): return self.maximum(0.0)
  def sigmoid(self): return 0.5 * (1.0 + (0.5 * self).tanh())
  # bools
  def __lt__(self, y): return _binary(Op.CMPLT, self, y)
  def __gt__(self, y): return _binary(Op.CMPLT, y, self)
  def where(self, x, y, dtype="float32"):
    dtype = x.dtype if isinstance(x, Tracer) else y.dtype if isinstance(y, Tracer) else dtype
    x, y = _lift(x, dtype), _lift(y, dtype)
    shape = broadcast_shapes(self.shape, x.shape, y.shape)
    return Tracer(node(Op.WHERE, *[t.broadcast_to(shape).node for t in (self, x, y)]))
  def __bool__(self):
    raise TypeError("traced value does not carry data at all, you can't use it in statements or elsewhere. Use where()")
  # reducts, shape ops, matmul
  def permute(self, *axes): return Tracer(node(Op.PERMUTE, self.node, attr=axes))
  @property
  def T(self): return self.permute(*reversed(range(self.ndim)))
  def _reduce(self, op, axis, keepdims):
    # we asked for reduce to axis = 1
    axes = range(self.ndim) if axis is None else (axis,) if isinstance(axis, int) else axis
    axes = tuple(sorted(a + self.ndim if a < 0 else a for a in axes))
    out = Tracer(node(op, self.node, attr=axes))
    return out if keepdims else out.reshape([n for i, n in enumerate(out.shape) if i not in axes])
  def sum(self, axis=None, keepdims=False): return self._reduce(Op.SUM, axis, keepdims)
  def max(self, axis=None, keepdims=False): return self._reduce(Op.RMAX, axis, keepdims)
  def mean(self, axis=None, keepdims=False):
    total = self.sum(axis, keepdims)
    return total * (1.0 / (prod(self.shape) // prod(total.shape)))
  def __matmul__(self, y):
    if self.ndim < 2 or y.ndim < 2: raise ValueError("matmul needs inputs with at least 2 dims")
    shapes = broadcast_shapes(self.shape[:-2], y.shape[:-2])
    x, y = self.broadcast_to(shapes + self.shape[-2:]), y.broadcast_to(shapes + y.shape[-2:])
    b = "abcdefgh"[:len(shapes)]
    return Tracer(node(Op.EINSUM, x.node, y.node, attr="{0}ij,{0}jk->{0}ik".format(b)))

def _lift(x, dtype="float32") -> Tracer:
  if isinstance(x, Tracer): return x
  if isinstance(x, (bool, int, float)): return Tracer(const(x, dtype))
  raise TypeError("can't use {} in a traced function".format(type(x).__name__))

def broadcast_shapes(*shapes):
  ndim = max(len(s) for s in shapes)
  out = []
  for sizes in zip(*[(1,) * (ndim - len(s)) + s for s in shapes]):
    big = set(sizes) - {1}
    if len(big) > 1: raise ValueError("can't broadcast shapes {}".format(list(shapes)))
    out.append(big.pop() if big else 1)
  return tuple(out)

def _binary(op, x, y) -> Tracer:
  dtype = x.dtype if isinstance(x, Tracer) else y.dtype
  x, y = _lift(x, dtype), _lift(y, dtype)
  shape = broadcast_shapes(x.shape, y.shape)
  return Tracer(node(op, x.broadcast_to(shape).node, y.broadcast_to(shape).node))

def _unary_dim(op, x, shape) -> Tracer:
  if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
  return Tracer(node(op, x.node, attr=tuple(shape)))

# this fnx will return the end result node on function run tracer
def trace(f, *inputs: Node) -> Node:
  return _lift(f(*[Tracer(v) for v in inputs])).node

