from contextlib import contextmanager
import numpy as np


# as the graph walking for counting gradients is the separate algorithm I'm moving
# that calculation here. decided early before it grows up
# it strips down recursive topowalk in Tensor so it is safer

_enabled = True
def is_enabled() -> bool: return _enabled

@contextmanager
def no_grad():
    global _enabled
    previous, _enabled = _enabled, False
    try: yield
    finally: _enabled = previous

def toposort(root):
    order, seen, stack = [], set(), [(root, False)]
    while stack:
        node, inputs_done = stack.pop()
        if inputs_done: order.append(node); continue
        if id(node) in seen: continue
        seen.add(id(node))
        stack.append((node, True))
        if node.grad_fn is not None: stack.extend((p, False) for p in node.grad_fn.parents)
    return order

# this will return gradients of root with respect to every leaf in
# the graph with requires_grad=True as a list of (leaf, ndarray)
def backward(root, grad=None):
    if not root.requires_grad:
        raise RuntimeError("backward() is called on non required grad tensor")
    if grad is None:
        if root.data.size != 1:
            raise RuntimeError("backward() without a gradient requires one-element output, instead got shape {}".format(root.data.shape))
        grad = np.ones_like(root.data)
    grads, leaves = {id(root): grad}, []
    for node in reversed(toposort(root)):
        g = grads.pop(id(node), None)
        if g is None: continue
        if node.grad_fn is None: leaves.append((node, g)); continue
        for parent, pg in zip(node.grad_fn.parents, node.grad_fn.backward(g)):
            if pg is None or not parent.requires_grad: continue
            if pg.shape != parent.data.shape:
                raise RuntimeError("{} should return shape {}, instead got shape {}".format(node.grad_fn, parent.data.shape, pg.shape))
            grads[id(parent)] = grads[id(parent)] + pg if id(parent) in grads else pg
    return leaves


