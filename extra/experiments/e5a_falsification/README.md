# E5a: Boundary-Localized Semantic Divergence under Falsification

> Not part of the paper reproduction; preserved as supplementary evidence.

## Purpose

Show that semantic divergence among tool-compatible STL backends (Breach, RTAMT,
RTAMT-dense, py-metric-temporal-logic, TaLiRo, STLCG++) is *boundary-localized*:
on random traces the backends agree on the falsification verdict (~100%), but
falsification search adversarially concentrates on the satisfaction boundary --
where the semantics diverge -- so the semantic choice can change falsification
efficacy. The size of the effect depends on the optimizer: rank-based (CMA-ES)
vs magnitude-based (simulated annealing, as in S-TaLiRo). `native` is the
backend-neutral dense oracle, not a condition.
