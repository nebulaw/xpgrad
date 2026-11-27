from xpgrad.graph import show, var
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

