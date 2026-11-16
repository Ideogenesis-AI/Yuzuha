# Spin

`Spin` labels an SU(2) irreducible representation by its spin quantum number
\(j \in \{0, \tfrac{1}{2}, 1, \tfrac{3}{2}, \ldots\}\). It is the atom the other core types
are built from: an [Edge](edge.md) pairs a `Spin` with a direction, a [CGSpec](cgspec.md)
is built from edges, and the dimension \(2j + 1\) a `Spin` reports fixes the shape of
every basis and symbol array derived from it.

::: yuzuha.Spin
    options:
      show_source: false
      heading_level: 2
      members:
        - value
        - twice
        - dimension
        - is_integer
        - is_half_integer

## Description

Spins are stored internally as the doubled integer \(2j\), so that no part of the library
performs floating-point arithmetic on a representation label. Comparisons, triangle-rule
checks, and cache keys are therefore all exact integer operations. The price is that the
constructor takes the **doubled** value in place of \(j\) itself:

```python
import yuzuha

j0    = yuzuha.Spin(0)   # j = 0
jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1
j32   = yuzuha.Spin(3)   # j = 3/2
```

`value()` converts back to a Python float where a human-readable \(j\) is wanted, and
`twice()` exposes the stored integer for code that needs to key on it. `is_integer()` and
`is_half_integer()` partition the spins by Frobenius-Schur class, which decides
the sign picked up on arrow reversal — see [Metric Tensor](../primitives/metric.md).

## See Also

- [Edge](edge.md): Combines a `Spin` with a `Direction`
- [CGSpec](cgspec.md): Specifies a full CG tensor from a list of `Edge` values
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `RepLabel`
  specification that `Spin` implements for SU(2)

## Notes

`Spin` values are immutable. Equality, hashing, and `<` ordering are all supported:
`Spin` objects can be used as dictionary keys and in sets, and lists of spins can be
passed to `sorted()` directly without a key function. The ordering is on the doubled
integer, which supplies the total order the protocol requires of a representation label.

The identity representation is `Spin(0)` (\(j = 0\)), which has dimension 1 and
contributes a trivial factor in all coupling coefficients. A negative doubled value
raises `ValueError`.
