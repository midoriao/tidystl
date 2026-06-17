from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tidystl import TorchSignal, evaluate, parse  # noqa: E402
from tidystl_compat import StlcgppTorchBackend  # noqa: E402


def test_stlcgpp_torch_backend_returns_tensor_and_supports_gradients() -> None:
    x = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
    signal = TorchSignal.from_dict(times=np.arange(3, dtype=float), values={"x": x})
    formula = parse("F[0,1](x >= 0)")

    result = evaluate(formula, signal, backend=StlcgppTorchBackend())
    assert isinstance(result.robustness, torch.Tensor)
    loss = result.robustness.sum()
    loss.backward()

    # Gradients accumulate on the leaf tensor; signal.values is a non-leaf
    # produced by torch.stack and never receives .grad.
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_stlcgpp_torch_backend_soft_approximation_is_differentiable() -> None:
    x = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
    signal = TorchSignal.from_dict(times=np.arange(3, dtype=float), values={"x": x})
    formula = parse("F[0,1](x >= 0)")

    exact = evaluate(formula, signal, backend=StlcgppTorchBackend()).robustness.detach().clone()
    soft = evaluate(
        formula,
        signal,
        backend=StlcgppTorchBackend(approx_method="softmax", temperature=2.0),
    ).robustness
    soft.sum().backward()

    assert x.grad is not None
    assert not torch.allclose(soft.detach(), exact)


def test_stlcgpp_torch_backend_trace_is_exposed() -> None:
    signal = TorchSignal.from_dict(
        times=np.arange(2, dtype=float),
        values={"x": torch.tensor([[1.0, -1.0]])},
    )
    formula = parse("x >= 0")

    result = evaluate(formula, signal, backend=StlcgppTorchBackend())
    trace = result.trace_for(formula)
    assert isinstance(trace, torch.Tensor)
    assert trace.shape == (1, 2)
