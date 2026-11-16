# CGSpec

`CGSpec` is the central data structure in Yuzuha: it identifies a CG tensor uniquely by
recording its external edges together with the internal spins of its fusion tree. Every
symbol function takes or returns one, and a spec doubles as the cache key under which the
corresponding basis and symbol arrays are stored.

::: yuzuha.CGSpec
    options:
      show_source: false
      heading_level: 2
      members:
        - from_edges
        - edges
        - num_external
        - om_dimension
        - with_inverted_axes

## Description

A spec has two parts:

- The **external edges** — a list of [Edge](edge.md) values, one per external edge
- The **internal spins** (\(\alpha\) values) — one per intermediate fusion step in the
  left-associative fusion tree

For an \(n\)th-order tensor there are \(n - 2\) internal spins. You never supply them by
hand, and they are not exposed as a Python attribute: `from_edges` enumerates every
internal spin configuration admitted by the triangle inequality and retains them
internally, which makes the spec's outer multiplicity well defined. What you see
of them is their count, through `om_dimension()`.

### Constructor

```python
import yuzuha

jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1

# 3rd-order spec: two incoming spin-1/2, one outgoing spin-1
spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(j1),
])
print(spec.num_external())   # 3
print(spec.om_dimension())   # 1

# The external shape is derived from the edges
print(tuple(e.j.dimension() for e in spec.edges))   # (2, 2, 3)
```

### Key Properties

| Method / Attribute | Description |
|--------------------|-------------|
| `edges` | List of the external `Edge` values, in order |
| `num_external()` | Number of external edges |
| `om_dimension()` | Outer-multiplicity (OM) dimension — number of valid internal spin paths |
| `with_inverted_axes(axes)` | Returns a copy with selected edge directions flipped |

### Axis Inversion

`with_inverted_axes(axes)` creates a new `CGSpec` with the edge directions at the given
axis indices flipped (`Incoming ↔ Outgoing`). The internal spin configurations are
preserved, because they depend only on the spin values and not on the arrows:

```python
# Flip the direction of edge 0
spec_inv = spec.with_inverted_axes([0])

# Flip multiple edges
spec_inv2 = spec.with_inverted_axes([0, 2])
```

Tensor network algorithms use this when a contracted pair has to be re-expressed with its
arrows reversed. The method supplies the flipped spec; the caller is responsible for the
accompanying Frobenius-Schur phase \((-1)^{2j}\) per inverted bond, described under
[Metric Tensor](../primitives/metric.md).

## See Also

- [Edge](edge.md): Building block for `CGSpec`
- [Canonical Basis](../symbols/canonical-basis.md): Compute the canonical basis for a spec
- [X-Symbol](../symbols/xsymbol.md): Contract two `CGSpec` tensors
- [R-Symbol](../symbols/rsymbol.md): Permute the edges of a `CGSpec` tensor
- [Contraction](contraction.md): Specify which edges to contract
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `Spec` specification

## Notes

`CGSpec` objects are immutable and hashable, so they can be used as dictionary keys and
in sets. `with_inverted_axes` always returns a **new** object; the original is unchanged.

`from_edges` raises `ValueError` if the edge list is too short (fewer than 2 edges) or if
no valid internal spin configuration exists, which happens when the triangle inequality
fails for every candidate path.
