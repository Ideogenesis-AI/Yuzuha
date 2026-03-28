# Spin

Irreducible representation label for SU(2).

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

`Spin` represents an SU(2) irreducible representation by its spin quantum number
\(j \in \{0, \frac{1}{2}, 1, \frac{3}{2}, \ldots\}\). Internally, spins are stored
as doubled integers \(2j\) to avoid floating-point arithmetic throughout all
computations.

The constructor takes the **doubled** spin value as an integer argument:

```python
import yuzuha

j0    = yuzuha.Spin(0)   # j = 0
jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1
j32   = yuzuha.Spin(3)   # j = 3/2
```

### Properties

- `value()` — spin as a Python float (\(j = n/2\))
- `twice()` — internal doubled integer \(2j\)
- `dimension()` — representation dimension \(2j + 1\)
- `is_integer()` — True if \(j\) is a non-negative integer
- `is_half_integer()` — True if \(j\) is a strict half-integer

## See Also

- [Edge](edge.md): Combines a `Spin` with a `Direction`
- [CGSpec](cgspec.md): Specifies a full CG tensor from a list of `Edge` values
- [Yuzuha Protocol — Type System](../../protocol/types.md): Abstract `RepLabel`
  specification that `Spin` implements for SU(2)

## Notes

`Spin` values are immutable. Comparison and hashing are supported, so `Spin` objects
can be used as dictionary keys and in sets.

The identity representation is `Spin(0)` (\(j = 0\)), which has dimension 1 and
contributes a trivial factor in all coupling coefficients.
