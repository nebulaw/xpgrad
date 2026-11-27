import numpy as np

import xpgrad.nn as nn
from xpgrad import Tensor
from xpgrad.optim import Optimizer, SGD

# setup data
rng = np.random.default_rng(1337)
x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = Tensor([[0], [1], [1], [0]])

# a twinkle star model
model = nn.Sequential(
  nn.Linear(2, 8, gain=5/3, rng=rng),
  Tensor.tanh,
  nn.Linear(8, 1, rng=rng),
  Tensor.sigmoid
)
optim = SGD(model.params(), lr=1.0)

# now let's teach our twinkle star how to xor
for step in range(0, 2005):
  err = model(x) - y
  loss = (err * err).sum() / 4
  optim.zero_grad()
  loss.backward()
  optim.step()
  if step % 501 == 0: print(step, round(loss.item(), 5))

print((model(x).data > 0.5).astype(int).ravel())

