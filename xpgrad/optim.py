
class Optimizer:
  def __init__(self, params, rule):
    self.params, self.rule = list(params), rule
    self.state = [None] * len(self.params)
  def zero_grad(self):
    for p in self.params: p.grad = None
  def step(self):
    for i, p in enumerate(self.params):
      if p.grad is not None: p.data, self.state[i] = self.rule(p.data, p.grad.data, self.state[i])

def sgd(lr=0.001):
  def rule(p, g, state): return p - lr * g, state
  return rule

def momentum(lr=0.001, mu=0.9):
  def rule(p, g, v):
    v = g if v is None else mu * v + g
    return p - lr * v, v
  return rule


