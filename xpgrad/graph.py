from __future__ import annotations
from typing import Any, Tuple
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


