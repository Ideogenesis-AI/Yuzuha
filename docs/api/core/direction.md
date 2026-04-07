# Direction

Arrow direction for tensor network edges.

::: yuzuha.Direction
    options:
      show_source: false
      heading_level: 2

## Description

`Direction` is a two-valued enum that specifies whether an external edge is **incoming**
(primal representation) or **outgoing** (dual representation). Arrow directions govern
how charges fuse and how the invariant metric is applied when reversing an arrow.

```python
from yuzuha import Direction

d_in  = Direction.Incoming   # primal / ket-type index
d_out = Direction.Outgoing   # dual / bra-type index
```

In the sign convention used throughout Yuzuha:

- An **incoming** edge with spin \(j\) contributes a factor of \(+1\) to the total
  magnetic quantum number sum.
- An **outgoing** edge with spin \(j\) contributes a factor of \(-1\).

Charge conservation then requires that the total signed magnetic quantum number of all
edges is zero:

$$\sum_{\text{incoming}} m_i - \sum_{\text{outgoing}} m_j = 0$$

Arrow reversal from incoming to outgoing is implemented via the metric tensor \(g\).
Arrow reversal from outgoing to incoming uses \(g^{-1} = (-1)^{2j}\, g\), which is
\(+g\) for integer spins and \(-g\) for half-integer spins.

## See Also

- [Edge](edge.md): Pairs `Direction` with a `Spin`
- [Metric Tensor](../primitives/metric.md): `use_as_incoming`, `use_as_outgoing`
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `Direction` enum
  specification
- [Yuzuha Protocol — Arrow Conventions](../../protocol/conventions/arrows.md): Arrow
  reversal conventions

## Notes

`Direction` values are immutable — attribute assignment raises `AttributeError`.
Two `Direction` values compare equal when they represent the same orientation
(`Incoming == Incoming`, `Outgoing == Outgoing`). They can be passed directly to the
`Edge.incoming` / `Edge.outgoing` constructors.
