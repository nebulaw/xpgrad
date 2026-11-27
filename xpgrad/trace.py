from xpgrad.graph import Node, node, Op


# tracer records every op it has as a graph
class Tracer:
  def __init__(self, n: Node): self.node = n
  @property
  def shape(self): return self.node.shape
  @property
  def dtype(self): return self.node.dtype
  @property
  def ndim(self): return len(self.node.shape)
  def __repr__(self): return "Tracer({})".format(self.node)
  def __add__(self, x): return Tracer(node(Op.ADD, self.node, x.node))
  def __mul__(self, x): return Tracer(node(Op.MUL, self.node, x.node))

# this fnx will return the end result node on function run tracer
def trace(f, *inputs: Node) -> Node:
  return f(*[Tracer(v) for v in inputs]).node




