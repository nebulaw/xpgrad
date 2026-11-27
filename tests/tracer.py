from xpgrad.graph import show, var
from xpgrad.trace import trace

def foo(x, w): return x * w + x

a = var("a", (3,))
b = var("b", (3,))

print(show(trace(foo, a, b)))


