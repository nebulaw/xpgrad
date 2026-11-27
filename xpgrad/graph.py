from __future__ import annotations
from typing import Any, Tuple, List
from enum import Enum, auto
from math import prod
from dataclasses import dataclass

# this graph is for immutable nodes of prim ops.

class Op(Enum):
  VAR = auto(); CONST = auto() # leaves
  EXP = auto(); LOG = auto(); SQRT = auto(); TANH = auto(); RECIP = auto(); CAST = auto()
  ADD = auto(); MUL = auto(); MAX = auto(); CMPLT = auto(); WHERE = auto()
  SUM = auto(); RMAX = auto()
  RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto()
  EINSUM = auto()

# node knows its shape and dtype the moment it was created
@dataclass(frozen=True, eq=False, repr=False)
class Node:
  op: Op
  srcs: Tuple[Node, ...]
  attr: Any
  shape: Tuple[int, ...]
  dtype: str
  def __repr__(self): return "<{} {}{}>".format(self.op.name, self.dtype, list(self.shape))

def var(name, shape, dtype="float32") -> Node: return Node(Op.VAR, (), name, tuple(shape), dtype)
def const(value, dtype="float32") -> Node: return Node(Op.CONST, (), value, (), dtype)
def node(op: Op, *srcs: Node, attr=None) -> Node:
  _check_inputs(op, srcs)
  shape, dtype = infer[op](attr, *srcs)
  return Node(op, srcs, attr, shape, dtype)

def toposort(*roots: Node) -> List[Node]:
  order, seen, stack = [], set(), [(r, False) for r in reversed(roots)]
  while stack:
    n, srcs_done = stack.pop()
    if srcs_done: order.append(n); continue
    if n in seen: continue
    seen.add(n)
    stack.append((n, True))
    stack.extend((s, False) for s in reversed(n.srcs))
  return order

def show(*roots: Node) -> str:
  order = toposort(*roots)
  name = {n: "v{}".format(i) for i, n in enumerate(order)}
  return "\n".join("{} : {}{} = {}{} {}".format(
    name[n],
    n.dtype,
    list(n.shape),
    n.op.name.lower(),
    "" if n.attr is None else "[{}]".format(n.attr),
    " ".join(name[s] for s in n.srcs)).rstrip() for n in order
  )

# shape rules for the ops
def _same(attr, *xs):
  if len({(x.shape, x.dtype) for x in xs}) != 1:
    raise ValueError("inputs need equal shapes and dtypes, got {}".format(list(xs)))
  return xs[0].shape, xs[0].dtype

def _reduce(axes, x):
  if any(not 0 <= a < len(x.shape) for a in axes): raise ValueError("invalid axes {}, got shape {}".format(axes, x))
  return tuple(1 if i in axes else n for i, n in enumerate(x.shape)), x.dtype

def _expand(shape, x):
  if len(shape) != len(x.shape) or any(a != b and a != 1 for a, b in zip(x.shape, shape)):
    raise ValueError("expand only stretches axes 1, can't {} from {} to {}".format(x, x.shape, shape))
  return tuple(shape), x.dtype

def _reshape(shape, x):
  if any(s < 0 for s in shape): raise ValueError("can't reshape {} to {} due to negative axes".format(x.shape, shape))
  if prod(shape) != prod(x.shape): raise ValueError("can't reshape {} from {} to {}".format(x, x.shape, shape))
  return tuple(shape), x.dtype

def _permute(axes, x):
  if sorted(axes) != list(range(len(x.shape))): raise ValueError("{} is not a permutation for {}".format(axes, x))
  return tuple(x.shape[a] for a in axes), x.dtype

def _where(attr, c, x, y):
  if c.dtype != "bool" or c.shape != x.shape: raise ValueError("where needs a bool condition shaped like x")
  return _same(attr, x, y)

def _einsum(spec, *xs):
  ins, out = spec.split("->")
  if len(ins.split(",")) != len(xs): raise ValueError("{} needs {} inputs".format(spec, len(ins.split(","))))
  sizes = {}
  for letters, x in zip(ins.split(","), xs):
    if len(letters) != len(x.shape):
      raise ValueError("{} doesn't match {}".format(letters, x))
    for c, n in zip(letters, x.shape):
      if sizes.setdefault(c, n) != n: raise ValueError("axis {} is both {} and {}".format(c, sizes[c], n))
  if len(set(out)) != len(out) or len({x.dtype for x in xs}) != 1 or not set(out) <= set(sizes): raise ValueError("bad einsum {}".format(spec))
  return tuple(sizes[c] for c in out), xs[0].dtype

def _cast(dtype, x): return x.shape, dtype

def _cmplt(attr, x, y): return _same(attr, x, y)[0], "bool"

# TODO: safety check so no ops will get there, and for ARITY too
infer = {
  Op.EXP: _same, Op.LOG: _same, Op.TANH: _same, Op.RECIP: _same, Op.SQRT: _same, Op.CAST: _cast,
  Op.ADD: _same, Op.MUL: _same, Op.MAX: _same, Op.CMPLT: _cmplt, Op.WHERE: _where,
  Op.SUM: _reduce, Op.RMAX: _reduce,
  Op.RESHAPE: _reshape, Op.EXPAND: _expand, Op.PERMUTE: _permute,
  Op.EINSUM: _einsum,
}

# a bit of tinygrad style George is OG
# TODO: einsum for later
UNARY = {Op.EXP, Op.LOG, Op.TANH, Op.RECIP, Op.SQRT}
BINARY = {Op.ADD, Op.MUL, Op.MAX, Op.CMPLT}
NUMERIC = {Op.ADD, Op.MUL, Op.MAX, Op.SUM, Op.RMAX} # non-bools added
ARITY = {
  **{op: 1 for op in UNARY | {Op.CAST, Op.SUM, Op.RMAX, Op.EXPAND, Op.RESHAPE, Op.PERMUTE}},
  **{op: 2 for op in BINARY},
  Op.WHERE: 3
}

# checker for ops that was added without its own rules
_ops = set(Op) - {Op.VAR, Op.CONST}
_missing = (_ops - set(infer)) | (_ops - {Op.EINSUM} - set(ARITY))
if _missing: raise RuntimeError("ops without rules: {}".format(sorted(o.name for o in _missing)))

def _check_inputs(op, srcs):
  if op in ARITY and len(srcs) != ARITY[op]:
    raise ValueError("{} takes {} inputs and got {}".format(op.name, ARITY[op], len(srcs)))
  if op in UNARY and "float" not in srcs[0].dtype:
    raise ValueError("{} requires float input and got {}".format(op.name, srcs[0]))
  if op in NUMERIC and any(s.dtype == "bool" for s in srcs):
    raise ValueError("{} can't take bool input and got {}".format(op.name, list(srcs)))

