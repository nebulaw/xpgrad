from xpgrad.graph import Op, node, var, show

x, w = var("x"), var("w")
y = node(Op.ADD, node(Op.MUL, x, w), x)
print(y)
print(show(y))

