# Edge

A directed representation — one edge of a CG tensor.

::: yuzuha.Edge
    options:
      show_source: false
      heading_level: 2
      members:
        - incoming
        - outgoing

## Description

`Edge` combines a `Spin` and a `Direction` to describe a single external edge of a
CG tensor. All edges of a `CGSpec` are specified as a list of `Edge` values.

The two class-method constructors are the primary way to create edges:

```python
import yuzuha

j = yuzuha.Spin(1)   # j = 1/2

e_in  = yuzuha.Edge.incoming(j)   # spin-1/2 incoming edge
e_out = yuzuha.Edge.outgoing(j)   # spin-1/2 outgoing edge
```

### Fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `j` | `Spin` | Representation label |
| `dir` | `Direction` | Incoming or outgoing |

### Derived Quantities

- **Dimension**: `j.dimension()` — the number of magnetic quantum number states
  (\(2j + 1\))

## See Also

- [Spin](spin.md): Representation label
- [Direction](direction.md): Arrow direction enum
- [CGSpec](cgspec.md): Builds a full specification from a list of `Edge` values
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `Edge` specification

## Notes

`Edge` objects are immutable and hashable. They are created with the `incoming` and
`outgoing` constructors shown above, which take a `Spin`.

Arrow flipping (converting an incoming edge to an outgoing edge of the same spin, or
vice versa) requires applying the metric tensor — see
[Metric Tensor](../primitives/metric.md) — do not simply
swapping the direction field.
