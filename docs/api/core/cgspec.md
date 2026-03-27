# CGSpec

Full specification of a CG tensor: its external edges and internal fusion tree.

::: yuzuha.CGSpec
    options:
      show_source: false
      heading_level: 2
      members:
        - from_edges
        - num_external
        - om_dimension
        - shape
        - with_inverted_axes

## Description

`CGSpec` (Clebsch-Gordan specification) is the central data structure in Yuzuha. It
records everything needed to identify a CG tensor uniquely:

- The **external edges** — a list of `Edge` values, one per external edge
- The **internal spins** (\(\alpha\) values) — one per intermediate fusion step in
  the left-associative fusion tree

For an \(n\)th-order tensor there are \(n - 2\) internal spins. `from_edges` automatically
enumerates all valid internal spin configurations using the triangle inequality and
stores them in the `alphas` field.

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
print(spec.shape)            # (2, 2, 3)
```

### Key Properties

| Method / Attribute | Description |
|--------------------|-------------|
| `num_external()` | Number of external edges |
| `om_dimension()` | Outer-multiplicity (OM) dimension — number of valid internal spin paths |
| `shape` | Tuple of `(2j+1)` for each external edge |
| `with_inverted_axes(axes)` | Returns a copy with selected edge directions flipped |

### Axis Inversion

`with_inverted_axes(axes)` creates a new `CGSpec` with the edge directions at the
specified axis indices flipped (`Incoming ↔ Outgoing`). The internal spin configurations
(`alphas`) are preserved because they depend only on spin values:

```python
# Flip the direction of edge 0
spec_inv = spec.with_inverted_axes([0])

# Flip multiple edges
spec_inv2 = spec.with_inverted_axes([0, 2])
```

This is used in tensor network algorithms to account for Frobenius-Schur phases when
reversing contracted pairs; see the [0.1.3 changelog](../../getting-started/changelog.md)
for background.

## See Also

- [Edge](edge.md): Building block for `CGSpec`
- [Canonical Basis](../symbols/canonical-basis.md): Compute the canonical basis for a spec
- [X-Symbol](../symbols/xsymbol.md): Contract two `CGSpec` tensors
- [R-Symbol](../symbols/rsymbol.md): Permute the edges of a `CGSpec` tensor
- [Contraction](contraction.md): Specify which edges to contract
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `Spec` specification

## Notes

`CGSpec` objects are immutable after construction. The `with_inverted_axes` method
always returns a **new** object; the original is unchanged.

The `from_edges` constructor raises `ValueError` if the provided edges are inconsistent
(e.g. fewer than 2 edges) or if no valid internal spin configuration exists (triangle
inequality violated for all candidates).
