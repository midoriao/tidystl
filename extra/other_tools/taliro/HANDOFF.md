# TaLiRo ground-truth handoff

`generate_ground_truth.m` regenerates the TaLiRo robustness fixtures in
`packages/tidystl-compat/tests/taliro_ground_truth.jsonl` by running the real `dp_taliro` engine in MATLAB.
These pin TaLiRo's robustness semantics so a future `TaliroBackend`
(Python mimic) can be cross-validated, exactly as the Breach/RTAMT/STLCG++
fixtures are.

## Getting the S-TaLiRo toolbox

S-TaLiRo is distributed from its public Subversion repository. Check out
revision 146 (the revision these fixtures were pinned against):

```sh
svn checkout -r 146 https://subversion.assembla.com/svn/s-taliro_public/ s-taliro
```

The working copy contains `setup_taliro.m`, `dp_taliro/`, `auxiliary/`, and the
rest of the toolbox.

## Building dp_taliro (MATLAB mex)

Tested with MATLAB R2022b (9.13) and gcc 9.4; any MATLAB with a `mex`-configured C compiler
works (run `mex -setup C` first if needed).

In MATLAB, `cd` into the checked-out `s-taliro` folder and run `setup_taliro`
(it builds dp_taliro / dp_t_taliro / fw_taliro via `mex`). **`setup_monitor`
will fail** with
`monitor/on_line.c: fatal error: simstruc.h: No such file or directory` -- that
is the online **Simulink** monitor S-function and is irrelevant to offline
robustness. The `dp_taliro` mex builds before that failure, so this is fine;
alternatively skip `setup_taliro` and just build/`addpath` the `dp_taliro`
folder once the mex is built.

## Running the generator

In MATLAB:

```matlab
addpath('/path/to/s-taliro');
addpath('/path/to/s-taliro/dp_taliro');
addpath('/path/to/s-taliro/auxiliary');
cd('/path/to/tidystl');
generate_ground_truth
```

Or in one shot from a shell:

```sh
matlab -batch "addpath('/path/to/s-taliro'); addpath('/path/to/s-taliro/dp_taliro'); addpath('/path/to/s-taliro/auxiliary'); cd('/path/to/tidystl'); generate_ground_truth"
```

It writes `packages/tidystl-compat/tests/taliro_ground_truth.jsonl`, one JSON object
per case. `dp_taliro` returns a single scalar (robustness of the whole trace,
reported at t=0), so each record holds exactly one point at time 0; this lines
up with tidystl's top-level robustness `rho[:, 0]`.

## TaLiRo robustness semantics (pinned by live runs, R2022b)

- **Predicates are half-spaces `A*x <= b`.** A predicate `var >= c` over signal
  column `j` is `-x_j <= -c` (use `proj=j` to select the column).
- **Robustness is the Euclidean signed distance `(b - A*x)/||A||`** -- it
  NORMALIZES by `||A||`. For single-variable predicates `||A|| = 1`, so the
  value equals the raw signed distance and matches tidystl/Breach/RTAMT.
  Multi-variable predicates divide by `||A||`: the `norm_sum` / `norm_diff`
  cases (`x+y>=0`, `x-y>=1`) both yield `3/sqrt(2) = 2.1213...` rather than the
  raw `3`. **A tidystl-compatible backend must reproduce the `/||A||` factor.**
- **Timed operators reduce over sample points inside the window only**;
  `dp_taliro` does not interpolate the signal at window endpoints. E.g.
  `[]_[0,2]` on `x=[5 2 8 1 6]` (t=0..4) gives `min(5,2,8) = 2`.
- **Output is a scalar at t=0**, not a per-timestep trace (a second output
  `aux` carries internal automaton state, not robustness over time).

## Cases

29 cases. The 16 core cases: basic predicates (`predicate_gte`, `not_simple`,
`and_two`, `or_two`), always (`always_untimed/sliding/boundary`), eventually
(`eventually_untimed/sliding`), until (`until_basic/untimed`), nesting
(`nested_GF/FG`), non-uniform time (`nonuniform_always`), and the half-space
normalization divergence (`norm_sum`, `norm_diff`). Plus 13 edge cases pinning
the window semantics: empty windows (`always_empty_window`,
`eventually_empty_window`), past-end windows (`always_past_end`,
`eventually_past_end`, `always_big_window`, `eventually_big_window`), offset
windows (`always_offset`, `eventually_offset`), samples-only-vs-interpolation
probes (`interp_probe_alw`, `interp_probe_ev`), and until edges
(`until_past_end`, `until_never_sat`, `until_tight`). Each case carries its
tidystl-equivalent formula in a comment in `generate_ground_truth.m`.

The `*_untimed` cases use bounded intervals (`[0,4]`) rather than a true
untimed operator because tidystl's grammar requires explicit bounds; `[0,4]`
spans the whole 5-sample trace, so `rho(0)` matches the untimed value.
