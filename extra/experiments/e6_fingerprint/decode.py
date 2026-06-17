"""E6 exp 2 -- black-box semantics identification.

Given only a tool's per-timestep ROOT robustness on a battery of probes, recover
the tool's Core 6 configuration up to the identifiability quotient.

Method (joint signature decode over the enumerated config space):

1. A *battery* is a set of probes (formula + signal) chosen to activate every
   Core 6 axis (see ``battery.json``).
2. The *signature* of a config is, per probe, the root robustness array produced
   by ``GenericBackend(config)`` -- or the marker ``"RAISES:<ExceptionType>"`` if
   the evaluation raises. A tool's *observed* signature is the same, but read from
   the black-box ``evaluate(phi, signal, backend=tool)`` root output.
3. We enumerate the full coherent config space C (4*2*2*2*3*2 = 192 configs),
   compute each one's battery signature with ``generic``, and group configs by
   EQUAL signature. Each group is an observational-equivalence class; the set of
   groups is the identifiability quotient on this battery.
4. ``decode_config(observed)`` returns the equivalence class whose signature
   matches the observed one, plus ``per_axis``: for each axis, the value common to
   the whole class, or ``None`` if the class disagrees (the axis is unresolved).

All functions are pure and typed; the battery and the config space are the only
inputs.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tidystl_compat import GenericBackend, GenericConfig  # noqa: E402

from tidystl import Signal, evaluate, parse  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402


@cache
def _parse(formula_text: str) -> Node:
    """Memoized ``parse``. The signature/quotient machinery parses the same battery
    formulas hundreds of times across the 192-config enumeration, so caching removes
    parsing as the dominant cost of the experiment suite.

    Caching shares one ``Node`` AST across all callers. ``Node`` is not a frozen
    dataclass, but every consumer (the backends building/evaluating DAGs) treats the
    tree as read-only -- nothing mutates ``kind``/``children``/``attrs`` -- so the
    shared instance is safe. This is an invariant of convention, not of the type.
    """
    return parse(formula_text)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BATTERY_PATH = Path(__file__).resolve().parent / "battery.json"
SEMANTICS_SPACE_PATH = REPO_ROOT / "extra/experiments/registry/semantics_space.json"

TOOLS = ["native", "breach", "rtamt", "stlcgpp", "taliro", "pymtl"]

# The Core 6 axes and their options, in declaration order.
AXES: dict[str, tuple[str, ...]] = {
    "signal_model": ("pl_interp", "pl_samples", "zoh", "discrete"),
    "boundary": ("clamp", "pessimistic"),
    "terminal": ("none", "extend_penultimate"),
    "predicate": ("signed", "euclidean"),
    "equality": ("signed", "bigm", "epsilon"),
    "until_prefix": ("inclusive", "exclusive"),
}

# A per-probe signature entry is either a robustness array or a raises marker.
ProbeValue = NDArray[np.floating] | str
# A signature is one ProbeValue per probe, in battery order.
Signature = tuple[ProbeValue, ...]


@dataclass(frozen=True)
class Probe:
    """A single battery probe: a named (formula, signal) pair."""

    name: str
    formula_text: str
    signal: Signal


def load_battery(path: Path = BATTERY_PATH) -> list[Probe]:
    """Load the e6-owned battery file into a list of typed ``Probe`` objects.

    The order of probes is the JSON insertion order and is the canonical signature
    order used everywhere downstream.
    """
    raw: dict[str, Any] = json.loads(path.read_text())
    probes: list[Probe] = []
    for name, entry in raw.items():
        times = np.asarray(entry["times"], dtype=np.float64)
        values: dict[str, NDArray[np.floating]] = {
            k: np.asarray(v, dtype=np.float64) for k, v in entry["values"].items()
        }
        probes.append(
            Probe(name=name, formula_text=entry["formula"], signal=Signal.from_dict(times, values))
        )
    return probes


def _root_or_marker(robustness: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract the root (top-level) robustness trace from a backend result array.

    Backend robustness is shape ``(1, T)`` (one root row); we return that row.
    """
    arr = np.asarray(robustness, dtype=np.float64)
    if arr.ndim == 2:
        return arr[0]
    return arr


def signature(config: GenericConfig, battery: list[Probe]) -> Signature:
    """Compute the battery signature of ``config`` using the ``generic`` backend.

    Each probe contributes its root robustness array, or ``"RAISES:<Type>"`` if the
    evaluation raises.
    """
    backend = GenericBackend(config)
    out: list[ProbeValue] = []
    for probe in battery:
        phi = _parse(probe.formula_text)
        try:
            res = backend.evaluate(phi, probe.signal)
            out.append(_root_or_marker(res.robustness))
        except Exception as exc:  # noqa: BLE001 -- a raise is itself an observable signature
            out.append(f"RAISES:{type(exc).__name__}")
    return tuple(out)


def observed_signature(tool: str, battery: list[Probe]) -> Signature:
    """Compute a tool's black-box observed signature via ``evaluate(..., backend=tool)``."""
    out: list[ProbeValue] = []
    for probe in battery:
        phi = _parse(probe.formula_text)
        try:
            res = evaluate(phi, probe.signal, backend=tool)
            out.append(_root_or_marker(res.robustness))
        except Exception as exc:  # noqa: BLE001 -- a raise is itself an observable signature
            out.append(f"RAISES:{type(exc).__name__}")
    return tuple(out)


def _value_equal(a: ProbeValue, b: ProbeValue, atol: float = 1e-6) -> bool:
    """Robust equality of two probe values.

    - Two raise markers are equal iff the exception type matches.
    - A raise marker and an array are never equal.
    - Two arrays are equal iff same shape and elementwise close, treating matching
      +/-inf and matching NaN as equal.
    """
    a_is_str = isinstance(a, str)
    b_is_str = isinstance(b, str)
    if a_is_str or b_is_str:
        return a_is_str and b_is_str and a == b
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        return False
    return bool(np.allclose(a_arr, b_arr, atol=atol, rtol=0.0, equal_nan=True))


def signatures_equal(a: Signature, b: Signature, atol: float = 1e-6) -> bool:
    """Two signatures are equal iff equal probe-by-probe under ``_value_equal``."""
    if len(a) != len(b):
        return False
    return all(_value_equal(x, y, atol=atol) for x, y in zip(a, b, strict=True))


def enumerate_configs() -> list[GenericConfig]:
    """The full coherent Core 6 config space (Cartesian product of all axis options)."""
    configs: list[GenericConfig] = []
    for combo in itertools.product(*AXES.values()):
        configs.append(GenericConfig.from_dict(dict(zip(AXES.keys(), combo, strict=True))))
    return configs


@dataclass(frozen=True)
class EquivalenceClass:
    """An observational-equivalence class: configs sharing one battery signature."""

    signature: Signature
    members: tuple[GenericConfig, ...]


def equivalence_classes(battery: list[Probe]) -> list[EquivalenceClass]:
    """Partition the enumerated config space into equal-signature classes (the quotient)."""
    classes: list[EquivalenceClass] = []
    bucket_sigs: list[Signature] = []
    bucket_members: list[list[GenericConfig]] = []
    for cfg in enumerate_configs():
        sig = signature(cfg, battery)
        for i, existing in enumerate(bucket_sigs):
            if signatures_equal(existing, sig):
                bucket_members[i].append(cfg)
                break
        else:
            bucket_sigs.append(sig)
            bucket_members.append([cfg])
    for sig, members in zip(bucket_sigs, bucket_members, strict=True):
        classes.append(EquivalenceClass(signature=sig, members=tuple(members)))
    return classes


@dataclass(frozen=True)
class DecodeResult:
    """The decode of an observed signature.

    ``configs`` is the matched equivalence class (every config indistinguishable
    from the observation on this battery). ``per_axis`` maps each axis to the value
    common to the whole class, or ``None`` if the class disagrees on that axis
    (the axis is unresolved by this battery -- the quotient at work).
    """

    configs: tuple[GenericConfig, ...]
    per_axis: dict[str, str | None]

    def __contains__(self, config: GenericConfig) -> bool:
        return config in self.configs

    def __len__(self) -> int:
        return len(self.configs)


def _per_axis(configs: tuple[GenericConfig, ...]) -> dict[str, str | None]:
    """For each axis, the common value across ``configs``, or ``None`` if they disagree."""
    result: dict[str, str | None] = {}
    for axis in AXES:
        values = {getattr(cfg, axis) for cfg in configs}
        result[axis] = next(iter(values)) if len(values) == 1 else None
    return result


def decode_config(
    observed: Signature,
    battery: list[Probe],
    classes: list[EquivalenceClass] | None = None,
) -> DecodeResult:
    """Recover the equivalence class of configs matching ``observed`` on ``battery``.

    If ``classes`` is precomputed it is reused (the expensive step is enumerating the
    192 signatures); otherwise it is computed here.
    """
    if classes is None:
        classes = equivalence_classes(battery)
    for cls in classes:
        if signatures_equal(cls.signature, observed):
            return DecodeResult(configs=cls.members, per_axis=_per_axis(cls.members))
    # No class matched: the observation is outside the generic config space on this
    # battery. Return an empty result so callers see the failure explicitly.
    return DecodeResult(configs=(), per_axis=dict.fromkeys(AXES))


def documented_configs() -> dict[str, GenericConfig]:
    """Load each tool's documented Core 6 config from ``semantics_space.json``."""
    space: dict[str, Any] = json.loads(SEMANTICS_SPACE_PATH.read_text())
    tool_configs: dict[str, Any] = space["tool_configs"]
    return {tool: GenericConfig.from_dict(cfg) for tool, cfg in tool_configs.items()}


def main() -> None:
    """Print, for each of the six tools, the inferred per-axis config and class size."""
    battery = load_battery()
    classes = equivalence_classes(battery)
    docs = documented_configs()

    logger.info("E6 exp 2 -- black-box semantics identification")
    logger.info("battery probes (%d): %s", len(battery), ", ".join(p.name for p in battery))
    logger.info(
        "config space: %d configs -> %d observational-equivalence classes\n",
        sum(len(c.members) for c in classes),
        len(classes),
    )

    for tool in TOOLS:
        observed = observed_signature(tool, battery)
        result = decode_config(observed, battery, classes)
        doc = docs[tool]
        recovered = doc in result
        logger.info("tool=%s", tool)
        logger.info("  documented config in class: %s", recovered)
        logger.info("  equivalence-class size: %d", len(result))
        logger.info("  inferred per_axis:")
        for axis in AXES:
            inferred = result.per_axis[axis]
            doc_val = getattr(doc, axis)
            mark = "resolved" if inferred is not None else "UNRESOLVED (quotient)"
            match = "" if inferred is None else (" OK" if inferred == doc_val else " MISMATCH")
            logger.info(
                "    %-13s inferred=%-14s documented=%-18s [%s]%s",
                axis,
                inferred,
                doc_val,
                mark,
                match,
            )
        logger.info("")


if __name__ == "__main__":
    main()
