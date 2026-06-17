"""Generate STLCG++ ground-truth JSONL from the shared test cases.

``stlcgpp`` (and ``lark``) ship in the ``experiments`` dependency group, so
run this with:

    uv run --only-group experiments python extra/other_tools/stlcgpp/generate_ground_truth.py

It evaluates every case in ``packages/tidystl-compat/tests/_helpers/stlcgpp_cases.py`` with the real
STLCG++ engine and writes a single ``stlcgpp_ground_truth.jsonl`` file into
``packages/tidystl-compat/tests/``.

Notes:

    - The current fixtures intentionally reuse the RTAMT-style discrete cases.
    - Ground truth is generated with ``padding="last"`` because raw STLCG++
      default padding emits ``-1e9`` sentinels near end-of-trace, which is not
      a useful compatibility target for tidystl.
"""

from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

import numpy as np
import torch
from stlcgpp.formula import Always, And, Equal, Eventually, Negation, Or, Predicate, Until

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "tidystl" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "tidystl-compat"))

from tests._helpers.jsonl_io import write_jsonl  # noqa: E402
from tests._helpers.stlcgpp_cases import STLCGPP_CASES  # noqa: E402
from tidystl import parse  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402

OUT_FILE = REPO_ROOT / "packages" / "tidystl-compat" / "tests" / "stlcgpp_ground_truth.jsonl"
PADDING_MODE = "last"


def _predicate_for_var(name: str, index: int) -> Predicate:
    return Predicate(name, lambda signal, i=index: signal[:, i])


def _constant_predicate(value: float) -> Predicate:
    return Predicate(
        str(value),
        lambda signal, v=value: torch.full(
            (signal.shape[0],),
            v,
            dtype=signal.dtype,
            device=signal.device,
        ),
    )


def _torch_unary(name: str, inner: Predicate, fn) -> Predicate:
    return Predicate(name, lambda signal, pred=inner, op=fn: op(pred(signal)))


def _torch_binary(name: str, left: Predicate, right: Predicate, fn) -> Predicate:
    return Predicate(
        name,
        lambda signal, lhs=left, rhs=right, op=fn: op(lhs(signal), rhs(signal)),
    )


def _arith_to_stlcgpp(node: Node, var_indices: dict[str, int]) -> Predicate:
    match node.kind:
        case "var":
            name = node.attrs["name"]
            if not isinstance(name, str):
                raise TypeError("var node requires str name attr")
            index = var_indices.setdefault(name, len(var_indices))
            return _predicate_for_var(name, index)
        case "const":
            value = float(node.attrs["value"])
            return _constant_predicate(value)
        case "+":
            left, right = node.children
            return _arith_to_stlcgpp(left, var_indices) + _arith_to_stlcgpp(right, var_indices)
        case "-":
            left, right = node.children
            return _arith_to_stlcgpp(left, var_indices) - _arith_to_stlcgpp(right, var_indices)
        case "*":
            left, right = node.children
            return _arith_to_stlcgpp(left, var_indices) * _arith_to_stlcgpp(right, var_indices)
        case "/":
            left, right = node.children
            return _arith_to_stlcgpp(left, var_indices) / _arith_to_stlcgpp(right, var_indices)
        case "^":
            left, right = node.children
            lhs = _arith_to_stlcgpp(left, var_indices)
            rhs = _arith_to_stlcgpp(right, var_indices)
            return _torch_binary(f"pow({lhs.name}, {rhs.name})", lhs, rhs, torch.pow)
        case "abs":
            (child,) = node.children
            inner = _arith_to_stlcgpp(child, var_indices)
            return _torch_unary(f"abs({inner.name})", inner, torch.abs)
        case "sqrt":
            (child,) = node.children
            inner = _arith_to_stlcgpp(child, var_indices)
            return _torch_unary(f"sqrt({inner.name})", inner, torch.sqrt)
        case _:
            raise NotImplementedError(f"unsupported arithmetic node kind {node.kind!r}")


def _interval_indices(node: Node) -> list[int]:
    interval = node.attrs.get("interval")
    if not isinstance(interval, tuple) or len(interval) != 2:
        raise ValueError(f"{node.kind} node missing interval")
    start, end = interval
    return [int(start), int(end)]


def _formula_to_stlcgpp(node: Node, var_indices: dict[str, int]):
    match node.kind:
        case "predicate":
            left = _arith_to_stlcgpp(node.attrs["left"], var_indices)
            right = _arith_to_stlcgpp(node.attrs["right"], var_indices)
            diff = left - right
            op = node.attrs["op"]
            if op in (">=", ">"):
                return diff > 0.0
            if op in ("<=", "<"):
                return diff < 0.0
            if op == "==":
                return Equal(diff, 0.0)
            raise NotImplementedError(f"unsupported predicate operator {op!r}")
        case "not":
            return Negation(_formula_to_stlcgpp(node.children[0], var_indices))
        case "and":
            return And(
                _formula_to_stlcgpp(node.children[0], var_indices),
                _formula_to_stlcgpp(node.children[1], var_indices),
            )
        case "or":
            return Or(
                _formula_to_stlcgpp(node.children[0], var_indices),
                _formula_to_stlcgpp(node.children[1], var_indices),
            )
        case "always":
            return Always(
                _formula_to_stlcgpp(node.children[0], var_indices),
                interval=_interval_indices(node),
            )
        case "eventually":
            return Eventually(
                _formula_to_stlcgpp(node.children[0], var_indices),
                interval=_interval_indices(node),
            )
        case "until":
            return Until(
                _formula_to_stlcgpp(node.children[0], var_indices),
                _formula_to_stlcgpp(node.children[1], var_indices),
                interval=_interval_indices(node),
            )
        case _:
            raise NotImplementedError(f"unsupported STL node kind {node.kind!r}")


def _evaluate_case(formula_text: str, values: dict[str, tuple[float, ...]]) -> np.ndarray:
    var_indices: dict[str, int] = {}
    formula = _formula_to_stlcgpp(parse(formula_text), var_indices)
    var_names = sorted(var_indices, key=var_indices.get)
    signal = torch.tensor(
        np.stack([values[name] for name in var_names], axis=1),
        dtype=torch.float32,
    )
    return formula(signal, padding=PADDING_MODE).detach().cpu().numpy()


def main() -> None:
    records = []
    for case in STLCGPP_CASES:
        robustness = _evaluate_case(case.tidystl_formula, case.values)
        records.append(
            {
                "name": case.name,
                "time": [float(t) for t in case.times],
                "robustness": [float(r) for r in robustness],
            }
        )
    write_jsonl(OUT_FILE, records)
    print(f"wrote {OUT_FILE} ({len(records)} cases)")


if __name__ == "__main__":
    main()
