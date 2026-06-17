# TaLiRo diagnostic values (`taliro_diag.json`)

TaLiRo (dp_taliro) reports only a whole-trace scalar (robustness at t=0), so a choice is
observable only when the diagnostic case is built to make that choice affect the t=0 value.
`taliro_diag.json` holds dp_taliro's value per profile case (`pc_*`, keyed by the registry
case name); cases absent from it are TaLiRo-N/A in the profile (see `profile.py`).

Provenance (real dp_taliro, `mathworks/matlab:r2022b`; toolbox setup per
`extra/other_tools/taliro/HANDOFF.md`). Each `pc_*` case is a minimal example isolating
one implicit choice:

- `pc_signalmodel` (`G[0,2](x>=0)`, t=`[0,1,9]`, x=`[1,1,-9]`): **1.0**.
  `dp_taliro('[]_[0,2] p', ge1('p',1,0), [1;1;-9], [0;1;9])`. Dense/real-time reading --
  the far t=9 sample is outside the real window `[0,2]`, so min over in-window samples = 1.
- `pc_boundary` (`F[2,4](x>=0)`, t=`[0,1]`, x=`[5,3]`): **-inf**.
  `dp_taliro('<>_[2,4] p', ge1('p',1,0), [5;3], [0;1])`. Pessimistic past-end rule -- the
  window `[2,4]` is entirely past the trace end, so the sup over an empty set is -inf.
- `pc_metric` (`x - y >= 1`, x=5, y=1): **2.1213...** = 3/sqrt(2).
  `dp_taliro('p', struct('str','p','A',[-1 1],'b',-1), [5 1], [0])`. Euclidean
  normalization by ||A|| = sqrt(2).
Cases NOT in the file (TaLiRo-N/A) and why:
- `pc_terminal` (robustness at horizon): a final-sample *reporting* convention; it never
  affects the whole-trace t=0 robustness, so no t=0-observable case exists.
- `pc_equality`: TaLiRo predicates are half-spaces; `x == 3` is inexpressible.
- `pc_window` (signal interpolation): TaLiRo makes NO interpolation choice. `dp_taliro`
  computes a *robustness estimate of timed state sequences* (`dp_taliro.m` help; theory in
  Fainekos et al., ACC 2012) -- it reduces over the samples whose timestamps fall in a
  window and does not reconstruct the signal between them. S-TaLiRo *does* ship a
  `signal_interpolation/` module (PWL/pchip/pconst), but that generates the *input* test
  signals during falsification; it is unrelated to the robustness computation. So TaLiRo
  has no PWL/PWC answer for this axis. For the record we did measure it --
  `dp_taliro('[]_[0.5,1.5] p', struct('str','p','A',-1,'b',0), [-1;10;-1], [0;2;4])`
  returns **+inf** -- but that is the vacuous-empty-window artifact (the off-grid window
  `[0.5,1.5]` contains no sample, so bounded `always` is vacuously true), not a meaningful
  robustness value, so it is left out of the table (cell `--`).

Regenerate: set up dp_taliro per HANDOFF.md, then run the three one-liners above.
