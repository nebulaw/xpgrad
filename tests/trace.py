from xpgrad.graph import show, toposort, var
from xpgrad.trace import trace

def foo(x, w): return x * w + x

a = var("a", (3,))
b = var("b", (3,))

print(show(trace(foo, a, b)))

print("-" * 24 * 2)
def bar(x, b): return 2.0 * x + b
print(show(trace(bar, var("x", (2,3)), var("b", (3,)))))

print("-" * 24 * 2)
print(show(trace(lambda x: x.sigmoid(), var("z", (3,)))))
print("-" * 24 * 2)
print(show(trace(lambda x, y: x - y, var("w", (3,)), var("y", (3,)))))


print("-" * 24 * 2)
x = var("x", (3,))
print(show(trace(lambda x: (x > 0.0).where(x, 0.0), x)))

def baz(x):
  if x > 0.0: return x
  return -x

try: trace(baz, x)
except TypeError as e: print("TypeError", e)


def loss(x, y, w1, b1, w2, b2):
  h = (x @ w1 + b1).tanh()
  err = (h @ w2 + b2).sigmoid() - y
  return (err * err).mean()

x = var("x", (4, 2))
y = var("y", (4, 1))
w1 = var("w1", (2, 8))
b1 = var("b1", (8,))
w2 = var("w2", (8, 1))
b2 = var("b2", (1,))
out = trace(loss, x, y, w1, b1, w2, b2)
print(out, len(toposort(out)))
print(show(trace(lambda x: x.sum(axis=1), var("x", (2, 3)))))

def chain(x):
  for _ in range(10_001): x = x.tanh()
  return x

print(len(toposort(trace(chain, var("x", (2,))))))

