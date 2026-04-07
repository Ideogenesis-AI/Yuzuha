# Contraction

Specification of which edges to contract between two CG tensors.

::: yuzuha.Contraction
    options:
      show_source: false
      heading_level: 2

## Description

`Contraction` records the pairs of edge indices to be contracted when calling
`compute_xsymbol`. Each pair \((i, k)\) indicates that edge \(i\) of the first
tensor (spec A) is contracted with edge \(k\) of the second tensor (spec B).

```python
import yuzuha

# Contract edge 2 of spec_a with edge 0 of spec_b
contraction = yuzuha.Contraction([2], [0])

# Contract multiple pairs
contraction2 = yuzuha.Contraction([2, 3], [0, 1])
```

### Fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `axes_a` | `list[int]` | Edge indices from spec A to contract |
| `axes_b` | `list[int]` | Edge indices from spec B to contract |

### Validation

A `Contraction` is **valid** if:

1. `axes_a` and `axes_b` have the same length
2. Each index in `axes_a` is in range `[0, spec_a.num_external())`
3. Each index in `axes_b` is in range `[0, spec_b.num_external())`
4. For every pair \((i, k)\): the spin of edge \(i\) in spec A equals the spin of edge
   \(k\) in spec B
5. For every pair \((i, k)\): the direction of edge \(i\) in spec A is opposite to the
   direction of edge \(k\) in spec B (one incoming, one outgoing)

Violations raise `ValueError` from `compute_xsymbol`.

## See Also

- [compute_xsymbol](../symbols/xsymbol.md): Uses `Contraction` to produce an X-symbol
- [CGSpec](cgspec.md): Input specification whose edges are contracted
- [Yuzuha Protocol — Symbol Interface](../../protocol/symbols.md): Abstract contraction
  interface specification

## Notes

`Contraction` objects are immutable, hashable, and can be reused across multiple
`compute_xsymbol` calls with compatible specs. They support `==` comparison and can
be used as dictionary keys or in sets.

The order of pairs in `axes_a` and `axes_b` matters: pair \((i, k)\) at position 0
is treated as the first contraction, which affects the internal fusion tree ordering of
the output spec \(C\).
