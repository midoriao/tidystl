"""E6 exp 3 -- automated probe synthesis.

tidySTL searches a pool of small (formula, signal) candidates and, using the
generic backend as oracle, selects a minimal battery whose induced
observational-equivalence quotient is as fine as the candidate pool allows.
This makes the diagnostic battery synthesized-and-certified rather than
hand-curated. See docs/design.md for the backend model.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tidystl_compat import GenericBackend, GenericConfig  # noqa: E402

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    AXES,
    Probe,
    ProbeValue,
    Signature,
    enumerate_configs,
    signatures_equal,
)
from tidystl import Signal, parse  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402


def _signal(times: list[float], **values: list[float]) -> Signal:
    return Signal.from_dict(
        np.asarray(times, dtype=np.float64),
        {k: np.asarray(v, dtype=np.float64) for k, v in values.items()},
    )


# A spread of small signals: uniform integer grid, a trace whose windows run
# past the end, a sub-integer-spacing grid, a non-uniform grid, an equality
# trace, and a multi-variable trace.
_SIGNALS: dict[str, Signal] = {
    "uniform5": _signal([0, 1, 2, 3, 4], x=[-1, 2, -3, 4, -5], y=[1, -1, 1, -1, 1]),
    "past_end": _signal([0, 1, 2], x=[-5, -1, 3], y=[5, 5, -3]),
    "subint": _signal([0, 1, 2, 3], x=[1, -2, 3, 1], y=[1, 1, -1, 1]),
    "nonuniform": _signal([0, 0.5, 1.5, 3, 4], x=[0, 3, 1, -2, 2], y=[1, -1, 1, -1, 1]),
    "eqtrace": _signal([0, 1, 2, 3, 4], x=[3, 2, 4, 3, 1], y=[1, 1, 1, 1, 1]),
    "multivar": _signal([0, 1, 2], x=[1, -0.5, 2], y=[1, -0.5, 2]),
}

# A spread of small formulas covering every axis's activation precondition.
_FORMULAS: list[str] = [
    "x >= 0",
    "x == 3",
    "x + y >= 0",
    "2 * x >= 1",
    "G[0,2](x >= 0)",
    "G[1,2](x >= 0)",
    "F[2,4](x >= 0)",
    "F[1,3](x >= 0)",
    "G[0,1](x > 0)",
    "G[0.5,1.5](x > 0)",
    "(x >= 0) and (y >= 0)",
    "(x >= 0) or (y >= 0)",
    "(x >= 0) U[2,4] (y >= 0)",
]


def candidate_probes() -> list[Probe]:
    """The full candidate pool: every (formula, signal) pair whose variables the
    signal provides. Probes whose formula needs a variable the signal lacks are
    skipped. Names are ``<formula-slug>@<signal>`` and unique.
    """
    probes: list[Probe] = []
    for fname, signal in _SIGNALS.items():
        available = set(signal.labels)
        for text in _FORMULAS:
            phi = parse(text)
            needed = _formula_vars(phi)
            if not needed <= available:
                continue
            slug = _slug(text)
            probes.append(Probe(name=f"{slug}@{fname}", formula_text=text, signal=signal))
    return probes


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "_" for c in text]
    return "".join(keep).strip("_")


def _formula_vars(node: Node) -> set[str]:
    """All variable names referenced anywhere in an STL/arith AST.

    ``ArithNode`` is a type alias for ``Node`` (one unified node type), so the
    structure is walked by ``kind``: a ``var`` node carries ``attrs["name"]``; a
    ``predicate`` node holds its two arithmetic operands in ``attrs["left"]`` and
    ``attrs["right"]`` (not in ``children``); every other node's operands are its
    ``children``.
    """
    found: set[str] = set()

    def walk(n: Node) -> None:
        if n.kind == "var":
            name = n.attrs.get("name")
            if isinstance(name, str):
                found.add(name)
        elif n.kind == "predicate":
            left, right = n.attrs["left"], n.attrs["right"]
            assert isinstance(left, Node) and isinstance(right, Node)
            walk(left)
            walk(right)
        for child in n.children:
            walk(child)

    walk(node)
    return found


def _probe_value(config: GenericConfig, phi: Node, signal: Signal) -> ProbeValue:
    """One probe's signature entry for one config (array or RAISES marker).

    Takes the already-parsed formula so the matrix build parses each probe once
    rather than once per config (192x fewer parses).
    """
    try:
        res = GenericBackend(config).evaluate(phi, signal)
        arr = np.asarray(res.robustness, dtype=np.float64)
        return arr[0] if arr.ndim == 2 else arr
    except Exception as exc:  # noqa: BLE001 -- a raise is an observable signature
        return f"RAISES:{type(exc).__name__}"


@dataclass(frozen=True)
class SignatureMatrix:
    """Cached per-(config, probe) signature values.

    ``configs[i]`` is the i-th enumerated config; ``values[probe_name][i]`` is that
    config's probe value. Built once; all downstream quotient computations read it.
    """

    configs: tuple[GenericConfig, ...]
    probe_names: tuple[str, ...]
    values: dict[str, list[ProbeValue]]

    @property
    def n_configs(self) -> int:
        return len(self.configs)

    @classmethod
    def build(cls, probes: list[Probe]) -> SignatureMatrix:
        configs = tuple(enumerate_configs())
        values: dict[str, list[ProbeValue]] = {}
        for probe in probes:
            phi = parse(probe.formula_text)
            values[probe.name] = [_probe_value(cfg, phi, probe.signal) for cfg in configs]
        return cls(
            configs=configs,
            probe_names=tuple(p.name for p in probes),
            values=values,
        )

    def signature_of(self, config_index: int, battery_names: list[str]) -> Signature:
        return tuple(self.values[name][config_index] for name in battery_names)


def quotient_classes(matrix: SignatureMatrix, battery_names: list[str]) -> list[list[int]]:
    """Group config indices by equal signature on ``battery_names`` (tolerant equality)."""
    groups: list[list[int]] = []
    group_sigs: list[Signature] = []
    for i in range(matrix.n_configs):
        sig = matrix.signature_of(i, battery_names)
        for g, existing in enumerate(group_sigs):
            if signatures_equal(existing, sig):
                groups[g].append(i)
                break
        else:
            group_sigs.append(sig)
            groups.append([i])
    return groups


def discriminates(matrix: SignatureMatrix, probe_name: str, axis: str) -> bool:
    """True if ``probe_name`` alone distinguishes some pair of configs that differ
    ONLY in ``axis`` -- i.e. the probe is sensitive to that axis.
    """
    axis_pos = list(AXES).index(axis)
    # Compare configs that are identical except for `axis`. Group configs by their
    # "other axes" key; within a group, if the probe value varies, it discriminates.
    by_rest: dict[tuple[str, ...], list[int]] = {}
    for i, cfg in enumerate(matrix.configs):
        coords = tuple(getattr(cfg, a) for a in AXES)
        rest = coords[:axis_pos] + coords[axis_pos + 1 :]
        by_rest.setdefault(rest, []).append(i)
    col = matrix.values[probe_name]
    for indices in by_rest.values():
        if len(indices) < 2:
            continue
        first = col[indices[0]]
        if any(not signatures_equal((first,), (col[j],)) for j in indices[1:]):
            return True
    return False


def find_axis_discriminators(matrix: SignatureMatrix) -> dict[str, str | None]:
    """For each axis, the name of some probe that discriminates it, or None."""
    result: dict[str, str | None] = {}
    for axis in AXES:
        found: str | None = None
        for name in matrix.probe_names:
            if discriminates(matrix, name, axis):
                found = name
                break
        result[axis] = found
    return result


def _refine_partition(
    matrix: SignatureMatrix, partition: list[list[int]], probe_name: str
) -> list[list[int]]:
    """Split each class by tolerant-equal value on ``probe_name``.

    Exact, by construction: two configs are signature-equal on ``battery + [p]`` iff
    they are equal on ``battery`` (same input class) AND equal on ``p``. So refining
    the partition for ``battery`` by one probe yields the partition for
    ``battery + [p]`` -- identical to ``quotient_classes`` on the longer battery, but
    it compares one probe value per pair instead of re-scanning the whole battery.
    """
    col = matrix.values[probe_name]
    refined: list[list[int]] = []
    for cls in partition:
        subgroups: list[tuple[ProbeValue, list[int]]] = []
        for i in cls:
            value = col[i]
            for rep, members in subgroups:
                if signatures_equal((rep,), (value,)):
                    members.append(i)
                    break
            else:
                subgroups.append((value, [i]))
        refined.extend(members for _, members in subgroups)
    return refined


def synthesize_battery(matrix: SignatureMatrix) -> list[str]:
    """Greedily select probes that most refine the quotient, then drop redundant
    ones. Returns the chosen probe names (a minimal separating family achieving the
    pool's finest quotient).

    The greedy step refines the chosen-prefix partition by one candidate probe at a
    time (``_refine_partition``) rather than recomputing each candidate's full
    quotient from scratch, which keeps the search comparison-cheap.
    """
    chosen: list[str] = []
    partition: list[list[int]] = [list(range(matrix.n_configs))]  # empty battery: 1 class
    current = 1
    remaining = list(matrix.probe_names)
    while True:
        best_name: str | None = None
        best_partition: list[list[int]] | None = None
        best_n = current
        for name in remaining:
            refined = _refine_partition(matrix, partition, name)
            if len(refined) > best_n:
                best_n = len(refined)
                best_name = name
                best_partition = refined
        if best_name is None or best_partition is None:
            break
        chosen.append(best_name)
        remaining.remove(best_name)
        partition = best_partition
        current = best_n
    # Minimize: drop any probe whose removal does not coarsen the quotient.
    minimized = list(chosen)
    for name in list(minimized):
        trial = [n for n in minimized if n != name]
        if len(quotient_classes(matrix, trial)) == current:
            minimized = trial
    return minimized


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    probes = candidate_probes()
    matrix = SignatureMatrix.build(probes)
    battery = synthesize_battery(matrix)
    discriminators = find_axis_discriminators(matrix)
    logger.info("candidate pool: %d probes", len(probes))
    logger.info("synthesized battery (%d probes):", len(battery))
    for name in battery:
        logger.info("  %s", name)
    logger.info("induced quotient: %d classes", len(quotient_classes(matrix, battery)))
    logger.info("per-axis discriminators:")
    for axis, name in discriminators.items():
        logger.info("  %-13s %s", axis, name)


if __name__ == "__main__":
    main()
