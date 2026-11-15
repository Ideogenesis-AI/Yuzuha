# Contraction

A `Contraction` names the pairs of external edges to be joined when two CG tensors are
contracted. It is the third argument of [compute_xsymbol](../symbols/xsymbol.md), and it
is validated against both input specs: the paired edges must carry the same spin and
opposite arrow directions, which makes the contraction a legal SU(2) pairing.

::: yuzuha.Contraction
    options:
      show_source: false
      heading_level: 2

## Description

Each pair \((i, k)\) indicates that edge \(i\) of the first tensor (spec A) is contracted
with edge \(k\) of the second tensor (spec B). The two index lists are given separately
and matched up position by position:

```python
import yuzuha

# Contract edge 2 of spec_a with edge 0 of spec_b
contraction = yuzuha.Contraction([2], [0])

# Contract multiple pairs
contractions = yuzuha.Contraction([2, 3], [0, 1])
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

Condition 1 is enforced by the constructor, since it needs nothing but the two lists.
The remaining conditions depend on the specs being contracted and are therefore checked
by `compute_xsymbol`, which raises `ValueError` on a violation.

## See Also

- [compute_xsymbol](../symbols/xsymbol.md): Uses `Contraction` to produce an X-symbol
- [CGSpec](cgspec.md): Input specification whose edges are contracted
- [Yuzuha Protocol — Symbol Interface](../../protocol/symbols.md): Abstract contraction
  interface specification

## Notes

`Contraction` objects are immutable and hashable, support `==` comparison, and can be
reused across any number of `compute_xsymbol` calls with compatible specs.

The order of the pairs matters. The pair at position 0 is treated as the first
contraction, which feeds through to the edge order and hence the internal fusion tree of
the output spec \(C\).
