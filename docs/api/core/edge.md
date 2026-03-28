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
| `spin` | `Spin` | Representation label |
| `direction` | `Direction` | Incoming or outgoing |

### Derived Quantities

- **Dimension**: `spin.dimension()` — the number of magnetic quantum number states
  (\(2j + 1\))

## See Also

- [Spin](spin.md): Representation label
- [Direction](direction.md): Arrow direction enum
- [CGSpec](cgspec.md): Builds a full specification from a list of `Edge` values
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `Edge` specification

## Notes

`Edge` objects are immutable and hashable. They can be created from `Direction` and
`Spin` separately or via the convenience constructors shown above.

Arrow flipping (converting an incoming edge to an outgoing edge of the same spin, or
vice versa) requires applying the metric tensor — see
[use_as_incoming / use_as_outgoing](../primitives/metric.md) — rather than simply
swapping the direction field.
