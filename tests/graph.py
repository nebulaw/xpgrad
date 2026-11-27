from xpgrad.graph import Op, node, var, show

x, b = var("x", (2, 3)), var("b", (3,))
bb = node(Op.EXPAND, node(Op.RESHAPE, b, attr=(1, 3)), attr=(2, 3))
s = node(Op.SUM, node(Op.ADD, x, bb), attr=(0,))
print(show(s))
print(show(node(Op.PERMUTE, x, attr=(1, 0))))
try: node(Op.EXPAND, b, attr=(2, 3))
except ValueError as e: print("ValueError", e)



