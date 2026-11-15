# Metric Tensor

The invariant metric makes arrow directions tractable: it is the transformation
that turns an index in a representation space into an index in its dual. Yuzuha applies it
automatically wherever a spec's arrows deviate from the canonical pattern, and the sign it
carries for half-integer spins is the Frobenius-Schur phase that surfaces in every
arrow-reversal and bond-inversion rule in the library.

## Description

The **invariant metric** \(g^{(j)}\) implements the canonical isomorphism between a
representation and its dual:

$$g^{(j)}_{m,\,m'} = (-1)^{j - m}\,\delta_{m,\,-m'}$$

This is a \((2j+1) \times (2j+1)\) real orthogonal matrix. Its defining properties are

$$g^{(j)} \cdot \left(g^{(j)}\right)^\top = I, \qquad
\left(g^{(j)}\right)^2 = (-1)^{2j}\, I,$$

so the metric is symmetric (\(g^\top = g\)) for integer spins and antisymmetric
(\(g^\top = -g\)) for half-integer ones, which can be written uniformly as
\(g^\top = (-1)^{2j} g\).

### Arrow Reversal

To convert an **incoming** edge to an **outgoing** one, contract with \(g\):

$$\tilde{T}^{m'}_{\ldots} = \sum_m g^{(j)}_{m, m'}\, T^{\ldots}_{m}$$

To go the other way, contract with the inverse:

$$\tilde{T}_{\ldots m'} = \sum_m \left(g^{(j)}\right)^{-1}_{m', m}\, T^{\ldots m}$$

The two directions are not interchangeable. Because \(g^{-1} = (-1)^{2j}\, g^\top\), they
differ by a sign for half-integer spins, and this is exactly the asymmetry a tensor
network algorithm has to account for when it reverses a bond.

### Frobenius-Schur Indicator

That recurring sign is the **Frobenius-Schur indicator**:

$$\text{FS}(j) = (-1)^{2j} = \begin{cases} +1 & j \in \mathbb{Z} \\ -1 & j \in \mathbb{Z} + \tfrac{1}{2} \end{cases}$$

It is \(+1\) for integer representations and \(-1\) for half-integer ones. In the X-symbol
bond inversion scheme, flipping a contracted pair of spin \(j\) multiplies the X-symbol by
\(\text{FS}(j)\); the same factor accumulates once per deviating edge in the conjugate phase.

### Python API

The metric itself is a Rust-level primitive (`use_as_incoming`, `use_as_outgoing` in
`yuzuha::primitives`) and is applied for you inside `canonical_basis`. What the Python
namespace exposes are the two derived quantities a caller actually needs:

| Function | Returns | Description |
|----------|---------|-------------|
| `fs_phase_for_spin(spin)` | `float` | Frobenius-Schur indicator \((-1)^{2j}\) |
| `compute_conjugate(spec)` | `(float, CGSpec)` | Conjugate phase and the arrow-reversed spec |

```python
import yuzuha

jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1

# Frobenius-Schur phase
phase_half = yuzuha.fs_phase_for_spin(jhalf)   # -1.0  (half-integer)
phase_int  = yuzuha.fs_phase_for_spin(j1)      # +1.0  (integer)

# Conjugation returns the accumulated phase alongside the reversed spec
spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(j1),
])
conj_phase, conj_spec = yuzuha.compute_conjugate(spec)
```

## See Also

- [Clebsch-Gordan Coefficients](cg.md): CG phases are the building blocks of the metric
- [Direction](../core/direction.md): The arrow direction that the metric transforms
- [Canonical Basis](../symbols/canonical-basis.md): Applies the metric to deviating edges
- [Yuzuha Protocol — Arrow Conventions](../../protocol/conventions/arrows.md): Full
  specification of the metric convention
- [Yuzuha Protocol — Duality Interface](../../protocol/duality.md): Required properties of
  `fs_phase` and `compute_conjugate`

## Notes

The metric \(g^{(j)}\) is **not** the Euclidean metric. It is the unique
SU(2)-invariant bilinear form on \(V_j\) up to a global phase, and substituting the
identity matrix for it produces arrow-reversal results that are silently incorrect,
though the shapes still match.
