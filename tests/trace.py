from xpgrad.graph import show, var
from xpgrad.trace import trace

def foo(x, w): return x * w + x

a = var("a", (3,))
b = var("b", (3,))

print(show(trace(foo, a, b)))

print("-" * 24 * 2)
def bar(x, b): return 2.0 * x + b
print(show(trace(bar, var("x", (2,3)), var("b", (3,)))))

