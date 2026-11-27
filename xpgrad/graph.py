from __future__ import annotations
from typing import Any, Tuple, List
from enum import Enum, auto
from dataclasses import dataclass

# this graph is for immutable nodes of prim ops.

class Op(Enum):
  VAR = auto()
  ADD = auto()
  MUL = auto()


@dataclass(frozen=True)
class Node:
  op: Op
  srcs: Tuple[Node, ...]
  attr: Any

def var(name) -> Node: return Node(Op.VAR, (), name)
def node(op: Op, *srcs: Node, attr=None) -> Node:
  return Node(op, srcs, attr)


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
  return "\n".join("{} = {}{} {}".format(
    name[n],
    n.op.name.lower(),
    "" if n.attr is None else "[{}]".format(n.attr),
    " ".join(name[s] for s in n.srcs)).rstrip() for n in order
  )


