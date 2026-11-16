# Edge

An `Edge` is a [Spin](spin.md) paired with a [Direction](direction.md), describing one
external edge of a CG tensor. Edges are the unit a [CGSpec](cgspec.md) is assembled from:
the ordered list of edges fixes both the external shape of every tensor built from the
spec and the fusion order of its internal tree.

::: yuzuha.Edge
    options:
      show_source: false
      heading_level: 2
      members:
        - incoming
        - outgoing

## Description

The two class-method constructors are the primary way to create edges, and take a `Spin`
object from a doubled integer:

```python
import yuzuha

j = yuzuha.Spin(1)   # j = 1/2

e_in  = yuzuha.Edge.incoming(j)   # spin-1/2 incoming edge
e_out = yuzuha.Edge.outgoing(j)   # spin-1/2 outgoing edge

print(e_in.j)              # 1/2
print(e_in.is_incoming())  # True
```

### Fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `j` | `Spin` | Representation label |
| `dir` | `Direction` | Incoming or outgoing |

The direction can also be tested with `is_incoming()` and `is_outgoing()`, which avoids
having to compare against a `Direction` value. The edge's dimension is not stored
separately: it is `j.dimension()`, the number \(2j + 1\) of magnetic quantum number
states, and it becomes one axis of the canonical basis array.

## See Also

- [Spin](spin.md): Representation label
- [Direction](direction.md): Arrow direction enum
- [CGSpec](cgspec.md): Builds a full specification from a list of `Edge` values
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `Edge` specification

## Notes

`Edge` objects are immutable and hashable, so they can be used as dictionary keys and in
sets. Two edges are equal only when both their spin and their direction agree.

Flipping an arrow — turning an incoming edge into an outgoing edge of the same spin, or
the reverse — is not just a matter of swapping the direction field. The tensor data must
also be contracted with the invariant metric; see
[Metric Tensor](../primitives/metric.md).
