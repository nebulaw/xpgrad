from xpgrad.graph import Op, node, var, show

x, w, b = var("x", (2, 3)), var("w", (2, 3)), var("b", (3,))
y = node(Op.MUL, x, w)
print(show(y))
try: node(Op.ADD, y, b)
except ValueError as e: print("ValueError", e)

