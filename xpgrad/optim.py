
class Optimizer:
  def __init__(self, params, lr):
    self.params, self.lr = list(params), lr
  def zero_grad(self):
    for p in self.params: p.grad = None
  def step(self):
    for i, p in enumerate(self.params):
      if p.grad is not None: p.data = self._step(i, p.data, p.grad.data)
  def _step(self, i, p, g): raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, params, lr=0.1, momentum=0.0):
    super().__init__(params, lr)
    self.momentum = momentum
    self.velocity = [0.0] * len(self.params)
  def _step(self, i, p, g):
    if self.momentum:
      g = self.velocity[i] = self.momentum * self.velocity[i] + g
    return p - self.lr * g

# simple adam

