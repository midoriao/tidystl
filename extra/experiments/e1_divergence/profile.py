"""E1 profile: diagnose each real tool's implicit choices from its output.

For each implicit choice we run ONE diagnostic case (constructed to vary only
that choice) on each real tool, read the tool's per-timestep robustness, and map
it to the option it reveals. The per-option prediction is analytic and is printed
alongside the result, so the attribution is auditable rather than asserted: the
reader sees both the predicted signatures and the observed values.

Real-tool outputs come from the same sources as the divergence matrix: Breach from
the committed MATLAB cache (``cache/breach_column``); RTAMT (discrete- and dense-time),
STLCG++, and py-metric-temporal-logic (``py-MTL``) from the column files produced by
``batch.sh`` (``eval_external_tools.py``, running each real library). Two rows use a
tidystl reference: the signal-model row (native + breach-compat) to separate dense
(real-time) from discrete (index), and the interpolation row (native = PWL) to separate
PWL from PWC; everything else classifies from the tool output alone. The signal-model
and interpolation axes are independent -- real Breach reads the far sample by real time
(dense) yet reconstructs the signal piecewise-linearly (PWL).

This is generic-backend-free: each choice is read off from one diagnostic case, not
from a parametrized config space.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import tidystl_compat  # noqa: E402

from tidystl import Signal, evaluate, parse, use  # noqa: E402

use(tidystl_compat)

REG = REPO_ROOT / "extra/experiments/registry"
CACHE = REPO_ROOT / "extra/experiments/e1_divergence/cache/breach_column"
COLUMNS = REPO_ROOT / "extra/outputs/e1_divergence/columns"
# TaLiRo (dp_taliro) reports only a whole-trace scalar (robustness at t=0). Where a choice
# can be made to affect that scalar (boundary, signal model, distance metric) we run a
# t=0-observable case and record dp_taliro's value here; choices that only act on the
# reported final sample (terminal-step) or that TaLiRo cannot express (equality) stay N/A.
TALIRO_DIAG = REPO_ROOT / "extra/experiments/e1_divergence/taliro_diag.json"

TOOLS = ("breach", "rtamt", "rtamt_dense", "stlcgpp", "pymtl", "taliro")

# Status of one (tool, case) observation. Three distinct "no value" reasons:
#   COVERAGE ("--")  : the tool cannot express the case (off-grid window for the discrete
#                      tools; TaLiRo equality; py-mtl equality / multivar arithmetic).
#   UNOBSERVABLE ("n/o"): the choice is real but invisible in the tool's whole-trace output
#                         (TaLiRo's scalar cannot see a final-sample / horizon convention).
#   EMPTY            : the tool ran but returned an empty signal -- a dense bounded operator
#                      whose window runs past the trace end (py-mtl truncates its output
#                      domain instead of padding); this IS the beyond-horizon behaviour.
COVERAGE = "n/a (coverage)"
UNOBSERVABLE = "n/o"
EMPTY = "empty (domain truncated)"
MISSING = "MISSING"


def _signals() -> dict[str, Any]:
    return json.loads((REG / "signals_div.json").read_text())


def _specs() -> dict[str, Any]:
    return json.loads((REG / "specs_div.json").read_text())


def _mk_signal(entry: dict[str, Any]) -> Signal:
    times = np.asarray(entry["times"], dtype=np.float64)
    values: dict[str, NDArray[np.floating]] = {
        k: np.asarray(v, dtype=np.float64) for k, v in entry["values"].items()
    }
    return Signal.from_dict(times, values)


def _fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def signal_desc(case: str) -> str:
    """Compact description of the case's input samples, e.g. ``x=[5,5,5,-99] @ t=[0,1,2,9]``."""
    sig = _signals()[case]
    parts = [f"{k}=[{','.join(_fmt_num(x) for x in v)}]" for k, v in sig["values"].items()]
    return "; ".join(parts) + " @ t=[" + ",".join(_fmt_num(t) for t in sig["times"]) + "]"


# Inline sparkline (TikZ) of a case's signal, drawn SCHEMATICALLY for legibility: samples
# are spaced EVENLY by index (not by real time), so a far sample (e.g. t=9) no longer crams
# the near ones; the real times are shown as labels on a time axis below, and a non-uniform
# gap is marked with a small zig-zag break glyph (visual spacing != real time there). Also:
# a labelled dashed zero baseline, per-sample value labels, one polyline + dots per variable
# (x=blue, y=red, z=green), and an orange ring on the readout sample(s) with a rho@t=k tag.
# Plain \draw/\fill/\node -- needs only \usepackage{tikz} (no decorations library).
_SPARK_COLORS = {"x": "blue", "y": "red!70!black", "z": "green!55!black"}
_SPARK_STEP, _SPARK_H = 0.85, 1.1  # cm: x-spacing between samples; signal height


def _zigzag(a: float, b: float, y: float, amp: float = 0.05, teeth: int = 3) -> str:
    """A small zig-zag polyline from (a,y) to (b,y) -- a plain-TikZ axis-break glyph
    marking a non-uniform time gap (real time jumps here; visual spacing does not)."""
    n = teeth * 2
    pts = [(a, y)]
    pts += [(a + (b - a) * i / n, y + (amp if i % 2 else -amp)) for i in range(1, n)]
    pts.append((b, y))
    return (
        "\\draw[gray!55,line width=0.5pt] "
        + " -- ".join(f"({px:.3f},{py:.3f})" for px, py in pts)
        + ";"
    )


def signal_tikz(case: str, readout: int = 0) -> str:
    sig = _signals()[case]
    t = [float(x) for x in sig["times"]]
    vals = {k: [float(x) for x in v] for k, v in sig["values"].items()}
    n = len(t)
    allv = [x for v in vals.values() for x in v] + [0.0]
    vmin, vmax = min(allv), max(allv)
    vspan = (vmax - vmin) or 1.0

    xs = [i * _SPARK_STEP for i in range(n)]  # even spacing by index (schematic)
    w = xs[-1] if xs else 0.0

    def y_cm(vv: float) -> float:
        return (vv - vmin) / vspan * _SPARK_H

    # A gap is a "break" when its real-time span is well above the smallest gap (non-uniform).
    gaps = [t[i + 1] - t[i] for i in range(n - 1)]
    gmin = min(gaps) if gaps else 1.0
    is_break = [g > 1.5 * gmin + 1e-9 for g in gaps]

    z = y_cm(0.0)
    ay = -0.16  # time-axis height (below the box)
    p = [r"\begin{tikzpicture}[baseline={(0,0.42cm)},line width=0.9pt,line cap=round]"]
    # dashed zero baseline + "0" label (only when the baseline is strictly interior)
    p.append(
        f"\\draw[gray!55,dashed,line width=0.3pt] (-0.05,{z:.3f}) -- ({w + 0.05:.2f},{z:.3f});"
    )
    if vmin < 0 < vmax:
        p.append(f"\\node[font=\\scriptsize,gray,anchor=east] at (-0.08,{z:.3f}) {{0}};")
    # time axis below: a segment between each pair of dots (zig-zag where the gap is a break),
    # then a tick + real-time label under every sample.
    for i in range(n - 1):
        if is_break[i]:
            p.append(_zigzag(xs[i], xs[i + 1], ay))
        else:
            p.append(
                f"\\draw[gray!50,line width=0.4pt] ({xs[i]:.3f},{ay:.3f}) -- ({xs[i + 1]:.3f},{ay:.3f});"
            )
    for xi, tt in zip(xs, t, strict=True):
        p.append(
            f"\\draw[gray!55,line width=0.5pt] ({xi:.3f},{ay:.3f}) -- ({xi:.3f},{ay - 0.08:.3f});"
        )
        p.append(
            f"\\node[font=\\scriptsize,gray,anchor=north] at ({xi:.3f},{ay - 0.07:.3f}) {{{_fmt_num(tt)}}};"
        )
    p.append(f"\\node[font=\\scriptsize,gray,anchor=west] at ({w + 0.1:.2f},{ay:.3f}) {{$t$}};")
    # signal polyline(s) + dots + per-sample value labels (de-duplicated across variables).
    labelled: set[tuple[float, float]] = set()
    for k, v in vals.items():
        c = _SPARK_COLORS.get(k, "black")
        pts = [(xi, y_cm(vv)) for xi, vv in zip(xs, v, strict=True)]
        p.append(f"\\draw[{c}] " + " -- ".join(f"({px:.3f},{py:.3f})" for px, py in pts) + ";")
        for (px, py), vv in zip(pts, v, strict=True):
            p.append(f"\\fill[{c}] ({px:.3f},{py:.3f}) circle (1.5pt);")
            key = (round(px, 2), round(py, 2))
            if key in labelled:
                continue
            labelled.add(key)
            # Always label ABOVE the dot: a label below a floor dot (value == vmin) would
            # collide with the time-axis tick beneath it; above the dot is clear (the line
            # leaves a floor dot sideways, not upward).
            p.append(
                f"\\node[font=\\tiny,{c},anchor=south] at ({px:.3f},{py + 0.05:.3f}) {{{_fmt_num(vv)}}};"
            )
    # readout marker: ring the sample(s) at the readout index + a rho@t=k tag above them.
    rx = xs[readout]
    ytop = max(y_cm(v[readout]) for v in vals.values())
    for v in vals.values():
        p.append(
            f"\\draw[orange!85!black,line width=0.7pt] ({rx:.3f},{y_cm(v[readout]):.3f}) circle (2.6pt);"
        )
    p.append(
        f"\\node[font=\\scriptsize,orange!80!black,anchor=south] at ({rx:.3f},{ytop + 0.22:.2f}) {{$\\rho$@$t{{=}}{_fmt_num(t[readout])}$}};"
    )
    p.append(r"\end{tikzpicture}")
    return "".join(p)


def observed(tool: str, case: str) -> list[float] | str:
    """The tool's diagonal robustness on ``case`` (spec==signal), or a status string."""
    if tool == "breach":
        p = CACHE / f"{case}.csv"
        if not p.exists():
            return MISSING
        rows = list(csv.reader(p.read_text().splitlines()))[1:]
        return [float(r[1]) for r in rows]
    f = COLUMNS / f"tool_{tool}_div.json"
    if not f.exists():
        return MISSING
    data = json.loads(f.read_text())
    for r in data["results"]:
        if r["spec"] == case and r["signal"] == case:
            if r["status"] != "ok":
                return COVERAGE
            if "rows" in r:
                # Dense tools (rtamt_dense, py-mtl) emit PWC breakpoints, the first at t=0;
                # an empty list is a bounded operator truncated past the trace end.
                if not r["rows"]:
                    return EMPTY
                return [float(v) for _, v in r["rows"]]
            return [float(v) for v in r["robustness"]]
    return MISSING


def taliro_value(case: str) -> list[float] | None:
    """TaLiRo's whole-trace robustness for ``case`` (from dp_taliro), or None if the
    choice is not observable in a t=0 scalar (terminal-step) or inexpressible (equality)."""
    diag: dict[str, list[float]] = json.loads(TALIRO_DIAG.read_text())
    return diag.get(case)


def _tidystl_diag(backend: str, case: str) -> list[float]:
    specs, signals = _specs(), _signals()
    phi = parse(specs[case]["tidystl"])
    rho = evaluate(phi, _mk_signal(signals[case]), backend=backend).robustness
    return [float(v) for v in np.asarray(rho)[0]]


# --- per-choice classifiers (analytic; the prediction note is shown in output) ---
#
# Each classifier receives either the observed diagonal (list[float]) or a status
# string (COVERAGE / MISSING). Non-``ok`` observations pass through unchanged, except
# the signal-model row, where a non-uniform-grid rejection IS the discrete behavior.


def cls_boundary(obs: list[float] | str) -> str:
    # F[2,4] window entirely past the trace end. Three observed behaviours:
    #   truncate     -> empty output (py-mtl drops the undefined region; no value past horizon);
    #   pessimistic  -> -inf (sup over an empty in-window set);
    #   clamp/extend -> the last value persists (held / extended over the past-end window).
    if obs == EMPTY:
        return "truncate (past horizon)"
    if isinstance(obs, str):
        return obs
    return "pessimistic" if any(math.isinf(v) and v < 0 for v in obs) else "clamp / extend-last"


def cls_terminal(obs: list[float] | str) -> str:
    if isinstance(obs, str):
        return obs
    # (x>=0) and (y>=0), violated only at final sample: exact -> -3; copy-prev -> +5.
    return "copy-prev" if obs[-1] > 0 else "exact"


def cls_equality(obs: list[float] | str) -> str:
    if isinstance(obs, str):
        return obs
    # x == 3: big-M -> +/-1e4 magnitudes; signed -> -|x-3| (small magnitudes).
    return "big-M (+/-1e4)" if max(abs(v) for v in obs) >= 1000 else "signed (-|f|)"


def cls_signal_model(obs: list[float] | str) -> str:
    # Time model: real-time (dense) vs sample-index (discrete). On the non-uniform
    # x=[1,1,-9]@[0,1,9] with G[0,2](x>=0), a dense model places the far t=9 sample at real
    # time 9, outside the window [0,2], so it does not pull the min down (-> 1; an idealised
    # continuous-PWL min would read -0.25, also dense). A discrete model indexes by position,
    # so "t=9" sits at index 2 inside the window -> -9, or it rejects the non-uniform grid.
    # NB: this case does NOT separate PWL from PWC -- that is the interpolation row
    # (pc_window), where the window lies strictly between samples. Real Breach reads 1 here
    # (real-time windowing), yet is PWL there (1.75); the two axes are independent.
    if obs == COVERAGE:
        return "discrete (rejects non-uniform)"
    if isinstance(obs, str):
        return obs
    # Either dense reading (native continuous-PWL -0.25, or in-window real-time samples 1)
    # counts as dense; the t=0 scalar works for per-timestep tools and TaLiRo alike.
    dense_t0 = [
        _tidystl_diag("native", "pc_signalmodel")[0],
        _tidystl_diag("breach", "pc_signalmodel")[0],
    ]
    v = obs[0]
    if any(abs(v - d) <= 1e-6 for d in dense_t0):
        return "dense (real-time)"
    return "discrete (index)"


def cls_interp(obs: list[float] | str) -> str:
    # Signal reconstruction (PWL vs PWC), probed by an off-grid window. G[0.5,1.5](x>0) on
    # x=[-1,10,-1]@[0,2,4]: the window [0.5,1.5] lies strictly between the samples at t=0 and
    # t=2, so the tool must reconstruct x at the window edges. PWL interpolates linearly ->
    # x(0.5)=1.75; PWC holds the t=0 sample over [0,2) -> -1. Discrete tools cannot place an
    # off-grid window at all (--). This is the choice formerly logged as the window-endpoint
    # "residue": it is in fact the clean PWL/PWC axis, reproduced by native (PWL).
    # Sample-based tools (TaLiRo) make no reconstruction choice -- dp_taliro computes a
    # robustness estimate over the timed state sequence, so this axis does not apply (--);
    # see taliro_diag.README.md for the measured +inf empty-window artifact.
    if isinstance(obs, str):
        return obs
    pwl = _tidystl_diag("native", "pc_window")[0]  # tidystl PWL reference: 1.75
    return "piecewise-linear (PWL)" if abs(obs[0] - pwl) <= 1e-6 else "piecewise-constant (PWC)"


def cls_metric(obs: list[float] | str) -> str:
    # x - y >= 1 (single multivar predicate): signed -> raw margin; euclidean -> margin
    # divided by ||A|| = sqrt(2). Classify by the nearer of the two analytic predictions.
    if isinstance(obs, str):
        return obs
    signed = _tidystl_diag("native", "pc_metric")  # native uses the signed margin
    s = signed[0]
    e = s / math.sqrt(2.0)
    v = obs[0]
    return "signed" if abs(v - s) <= abs(v - e) else "euclidean"


@dataclass(frozen=True)
class Diagnostic:
    choice: str
    case: str
    prediction: str
    classify: Callable[[list[float] | str], str]
    # Timestep index whose robustness value is shown in the cell -- the point where the
    # choice manifests (0 = t0 for the t=0-observable rows; -1 = final sample for terminal).
    readout: int = 0
    # TaLiRo's no-value reason: "unobservable" (real choice, invisible in a scalar) or
    # "inexpressible" (cannot express the case); None when TaLiRo does observe the choice.
    taliro_na: str | None = None


DIAGNOSTICS: tuple[Diagnostic, ...] = (
    Diagnostic(
        "Signal model",
        "pc_signalmodel",
        "dense places t=9 at real time, outside window [0,2] (-> 1); discrete indexes by position, t=9 at index 2 inside (-> -9)",
        cls_signal_model,
    ),
    Diagnostic(
        "Signal interpolation",
        "pc_window",
        "off-grid window [0.5,1.5] between samples: PWL interpolates x(0.5)=1.75; PWC holds the t=0 sample (-1); discrete-index tools cannot place it and sample-based TaLiRo makes no reconstruction (--)",
        cls_interp,
    ),
    Diagnostic(
        "Signal beyond horizon",
        "pc_boundary",
        "F[2,4] window entirely past the trace end: clamp/extend -> last value (3); pessimistic -> -inf; truncate -> empty",
        cls_boundary,
    ),
    Diagnostic(
        "Robustness at horizon",
        "pc_terminal",
        "violated only at the final sample: exact -> -3; copy-prev -> penultimate +5",
        cls_terminal,
        readout=-1,  # the divergence is at the final sample
        taliro_na="unobservable",  # final-sample convention, invisible to a whole-trace scalar
    ),
    Diagnostic(
        "Equality predicate",
        "pc_equality",
        "at x=2 (off the satisfying value): signed -> -1; big-M -> -10^4",
        cls_equality,
        readout=1,  # t=1 has x=2 (off the satisfying value), exposing the magnitude gap
        taliro_na="inexpressible",  # TaLiRo predicates are half-spaces; == not expressible
    ),
    Diagnostic(
        "Distance metric",
        "pc_metric",
        "signed -> raw margin (3); euclidean -> margin / ||A|| = 3/sqrt(2)",
        cls_metric,
    ),
)


def profile() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for d in DIAGNOSTICS:
        cells: dict[str, Any] = {}
        for tool in TOOLS:
            if tool == "taliro":
                tv = taliro_value(d.case)
                if tv is None:
                    opt = UNOBSERVABLE if d.taliro_na == "unobservable" else COVERAGE
                    cells[tool] = {"option": opt, "value": None}
                    continue
                cells[tool] = {"option": d.classify(tv), "value": tv[0]}
                continue
            obs = observed(tool, d.case)
            cells[tool] = {
                "option": d.classify(obs),
                "value": None if isinstance(obs, str) else obs[d.readout],
            }
        rows.append(
            {
                "choice": d.choice,
                "case": d.case,
                "signal": signal_desc(d.case),
                "readout": d.readout,
                "prediction": d.prediction,
                "tools": cells,
            }
        )

    return {"tools": list(TOOLS), "rows": rows}


def _fmt_val(v: float, latex: bool) -> str:
    if math.isinf(v):
        return (r"-\infty" if v < 0 else r"\infty") if latex else ("-inf" if v < 0 else "inf")
    if abs(v) >= 1000:  # big-M sentinel (+/-10^4)
        sign = "-" if v < 0 else ""
        return f"{sign}10^4" if latex else f"{sign}1e4"
    return f"{v:.3g}"


def _cell(cell: dict[str, Any], latex: bool) -> str:
    label = _disp(cell["option"]) if latex else cell["option"]
    v = cell.get("value")
    if v is None:
        return label
    tok = _fmt_val(v, latex)
    # option is primary; the robustness value is a small, de-emphasised parenthetical.
    return (
        f"{label}~{{\\scriptsize\\textcolor{{black!55}}{{$({tok})$}}}}"
        if latex
        else f"{label} ({tok})"
    )


def print_text(p: dict[str, Any]) -> None:
    w = 28
    print(f"{'choice':22}{'signal':34}" + "".join(f"{t:{w}}" for t in p["tools"]))
    for row in p["rows"]:
        line = f"{row['choice']:22}{row['signal']:34}"
        line += "".join(f"{_cell(row['tools'][t], latex=False):{w}}" for t in p["tools"])
        print(line)


# Short option labels for the table (full definitions live in the caption / prediction note).
_DISPLAY: dict[str, str] = {
    "clamp / extend-last": "clamp",
    "truncate (past horizon)": "truncate",
    "pessimistic": "pessimistic",
    "copy-prev": "copy-prev",
    "exact": "exact",
    "big-M (+/-1e4)": "big-M",
    "signed (-|f|)": "signed",
    "dense (real-time)": "dense",
    "piecewise-linear (PWL)": "PWL",
    "piecewise-constant (PWC)": "PWC",
    "discrete (index)": "discrete",
    "discrete (rejects non-uniform)": "discrete",
    "n/a (coverage)": r"\textemdash",
    "n/o": r"\textit{n/o}",
    "signed": "signed",
    "euclidean": "euclidean",
    "N/A": "N/A",
    "MISSING": "?",
}
_CHOICE_LABEL = {
    "breach": "Breach",
    "rtamt": "RTAMT",
    "rtamt_dense": "RTAMT$_d$",
    "stlcgpp": "STLCG++",
    "pymtl": "py-MTL",
    "taliro": "TaLiRo",
}


def _disp(option: str) -> str:
    return _DISPLAY.get(option, option)


def emit_latex(p: dict[str, Any]) -> str:
    specs = _specs()
    cols = "lll" + "l" * len(p["tools"])
    head = " & ".join(
        [
            "\\textbf{Implicit choice}",
            "\\textbf{Formula}",
            "\\textbf{Signal}",
            *[f"\\textbf{{{_CHOICE_LABEL[t]}}}" for t in p["tools"]],
        ]
    )
    lines = [
        "% AUTO-GENERATED by extra/experiments/e1_divergence/profile.py; do not hand-edit.",
        "% Requires \\usepackage{tikz}, booktabs. Signal sparkline (schematic): samples are",
        "% spaced EVENLY by index, not by real time; the t-axis below gives each sample's real",
        "% time, and a zig-zag break marks a non-uniform gap (visual spacing != real time there).",
        "% Dashed line = value 0; x=blue y=red z=green; orange ring = the readout sample whose",
        "% rho the cell reports (t=0, or the final sample = horizon for the robustness-at-horizon row).",
        "% Cell -- = tool cannot express the case (off-grid window for discrete tools;",
        "% TaLiRo equality; py-MTL equality / multivar arithmetic). n/o = real choice but",
        "% not observable in the tool's whole-trace output (TaLiRo). truncate = dense bounded",
        "% operator past the horizon returns an empty signal (py-MTL drops the region).",
        "% Interpolation row: PWL/PWC = linear-interpolate vs sample-and-hold at the off-grid",
        "% window edges; -- here = no reconstruction choice (discrete-index tools cannot place",
        "% an off-grid window; TaLiRo's robustness is a sample-based estimate, not interpolated).",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        head + " \\\\",
        "\\midrule",
    ]
    for row in p["rows"]:
        formula = specs[row["case"]]["tidystl"]
        cells = [_cell(row["tools"][t], latex=True) for t in p["tools"]]
        lines.append(
            f"{row['choice']} & \\texttt{{{formula}}} & {signal_tikz(row['case'], row['readout'])} & "
            + " & ".join(cells)
            + " \\\\[2pt]"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = profile()
    print_text(p)
    out_dir = REPO_ROOT / "extra/outputs/e1_divergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile.json").write_text(json.dumps(p, indent=2, default=lambda x: str(x)))
    (out_dir / "profile.tex").write_text(emit_latex(p))
    print(f"\nwrote {out_dir / 'profile.json'}")
    print(f"wrote {out_dir / 'profile.tex'}")


if __name__ == "__main__":
    main()
