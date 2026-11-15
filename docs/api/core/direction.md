# Direction

`Direction` is the two-valued enum recording whether an external edge is **incoming**
(primal representation) or **outgoing** (dual representation). Arrow directions are not
decoration: they decide how charges fuse, which magnetic-number sum has to vanish, and
where the invariant metric must be inserted when an arrow is reversed.

::: yuzuha.Direction
    options:
      show_source: false
      heading_level: 2

## Description

Every external edge of a CG tensor carries one of the two directions, built with the static
constructors or recovered from the `dir` field of an existing edge:

```python
from yuzuha import Direction

d_in  = Direction.incoming()   # primal / ket-type index
d_out = Direction.outgoing()   # dual / bra-type index

print(d_in.sign())             # 1
print(d_out.sign())            # -1
print(d_in.flip() == d_out)    # True
```

### Sign Convention

An incoming edge contributes \(+m\) to the total magnetic quantum number and an outgoing
edge contributes \(-m\). Charge conservation then requires the signed sum over all edges
to vanish, and any basis entry whose indices violate this is identically zero:

$$\sum_{\text{incoming}} m_i - \sum_{\text{outgoing}} m_j = 0$$

The rule survives arrow reversal because its two moving parts cancel. Reversing an edge
contracts it with the metric, which sends \(m \to -m\) on that edge, and at the same time
flips the sign that edge carries in the sum. The law therefore holds in the same form for
every arrow pattern, not merely the canonical one.

### Arrow Reversal

Flipping a direction is not a relabeling — the tensor data has to be transformed with the
invariant metric \(g\). Reversal from incoming to outgoing contracts with \(g\); reversal
from outgoing to incoming contracts with \(g^{-1} = (-1)^{2j}\, g\), which is \(+g\) for
integer spins and \(-g\) for half-integer spins. That sign is the Frobenius-Schur phase,
which justifies `Direction` as part of the cache key of every symbol.

## See Also

- [Edge](edge.md): Pairs `Direction` with a `Spin`
- [Metric Tensor](../primitives/metric.md): The transformation that reverses an arrow
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `Direction` enum
  specification
- [Yuzuha Protocol — Arrow Conventions](../../protocol/conventions/arrows.md): Arrow
  reversal conventions

## Notes

`Direction` values are immutable — attribute assignment raises `AttributeError`. Two
values compare equal when they represent the same orientation (`Incoming == Incoming`,
`Outgoing == Outgoing`), so they can be used as dictionary keys and compared directly
against the `dir` field of an `Edge`.
