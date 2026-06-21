# Specification Language

The formula grammar: operators, precedence, and predicate arithmetic. For a
guided introduction with worked examples, see the [user manual](usage.md).

## Operators

| Operator | Syntax | Node kind |
|---|---|---|
| Always (globally) | `G[a,b](phi)` | `always` |
| Eventually (finally) | `F[a,b](phi)` | `eventually` |
| Until | `phi U[a,b] psi` | `until` |
| Conjunction | `phi and psi` | `and` |
| Disjunction | `phi or psi` | `or` |
| Negation | `not phi` | `not` |

All temporal intervals `[a,b]` are bounded with `0 <= a <= b`; unbounded
operators are not supported. `G` and `F` require parentheses around their
argument: `G[0,5](x >= 0)`.

Binding strength, from loosest to tightest:

```text
or  <  and  <  U  <  not, G, F  <  atoms
```

So `not p and q` parses as `(not p) and q`, and `p and q or r` parses as
`(p and q) or r`. Operands of `U` are unary-level expressions; use
parentheses to put `and`/`or` under an until.

## Predicate arithmetic

Predicate operands are arithmetic expressions over variables and numeric
constants:

- binary: `+`, `-`, `*`, `/`, exponentiation `^`;
- unary: `-x` (negation), `abs(...)`, `sqrt(...)`;
- comparisons: `>=`, `>`, `<=`, `<`, `==`.

Robustness of an atomic predicate is its signed margin:

| Predicate | Robustness |
|---|---|
| `lhs >= rhs`, `lhs > rhs` | `lhs - rhs` |
| `lhs <= rhs`, `lhs < rhs` | `rhs - lhs` |
| `lhs == rhs` | `-abs(lhs - rhs)` |

Note two consequences of quantitative semantics: strict and non-strict
comparisons have identical robustness, and `==` never yields positive
robustness (0 at exact equality, negative otherwise).

The keywords `and`, `or`, `not`, `abs`, `sqrt` are reserved and cannot be
used as variable names.
