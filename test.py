from xpgrad import Tensor
import xpgrad.nn as nn
import numpy as np
import torch
from time import perf_counter

torch.manual_seed(1337)

def test_add():
    a = Tensor(np.random.rand(1000, 4000), requires_grad=True)
    b = Tensor(np.random.rand(4000, 2000), requires_grad=True)

    start = perf_counter()
    c = a.matmul(b)
    d = c.sum()
    end = perf_counter()
    print(f"XPGRAD-FWARD: {(end-start):.6f}")
    start = perf_counter()
    d.backward()
    end = perf_counter()
    print(f"XPGRAD-BWARD: {(end-start):.6f}")
    print(f"{a.shape=}, {a.grad.shape=}")
    print(f"{b.shape=}, {b.grad.shape=}")
    print(f"{c.shape=}, {c.grad.shape=}")
    print(f"{d.shape=}, {d.grad.shape=}")

    a_t = torch.tensor(a.data, requires_grad=True)
    b_t = torch.tensor(b.data, requires_grad=True)
    start = perf_counter()
    c_t = a_t.matmul(b_t)
    d_t = c_t.sum()
    end = perf_counter()
    print(f"TORCH--FWARD: {(end-start):.6f}")
    start = perf_counter()
    d_t.backward()
    end = perf_counter()
    print(f"TORCH--BWARD: {(end-start):.6f}")


    assert np.allclose(c.data, c_t.detach().numpy())
    assert np.allclose(a.grad.data, a_t.grad.numpy())
    assert np.allclose(b.grad.data, b_t.grad.numpy())
    print("ALL TRUE")

test_add()

