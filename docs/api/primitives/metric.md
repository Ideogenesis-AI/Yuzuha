# Metric Tensor

Arrow-reversal transformation for SU(2) representation spaces.

## Description

The **invariant metric** \(g^{(j)}\) implements the canonical isomorphism between a
representation and its dual. It allows converting an incoming edge (primal space) to an
outgoing edge (dual space) and vice versa, while correctly tracking the sign factor
required by SU(2) representation theory.

### Definition

$$g^{(j)}_{m,\,m'} = (-1)^{j - m}\,\delta_{m,\,-m'}$$

This is a \((2j+1) \times (2j+1)\) real orthogonal matrix. It satisfies:

$$g^{(j)} \cdot \left(g^{(j)}\right)^\top = I$$

$$\left(g^{(j)}\right)^2 = (-1)^{2j}\, I$$

For integer spins (\(j \in \mathbb{Z}\)) the metric is symmetric (\(g^\top = g\)). For
half-integer spins it is antisymmetric (\(g^\top = -g\)), or equivalently \(g^\top = (-1)^{2j} g\).

### Python API

The following functions are available in the Yuzuha Python namespace:

| Function | Description |
|----------|-------------|
| `fs_phase_for_spin(spin)` | Frobenius-Schur indicator \((-1)^{2j}\) |
| `compute_conjugate(spec)` | Canonical basis tensor with all arrows reversed |

```python
import yuzuha

jhalf = yuzuha.Spin(1)   # j = 1/2
j1    = yuzuha.Spin(2)   # j = 1

# Frobenius-Schur phase
phase_half = yuzuha.fs_phase_for_spin(jhalf)   # -1  (half-integer)
phase_int  = yuzuha.fs_phase_for_spin(j1)      # +1  (integer)

# Conjugated basis
spec = yuzuha.CGSpec.from_edges([
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.incoming(jhalf),
    yuzuha.Edge.outgoing(j1),
])
basis_conj = yuzuha.compute_conjugate(spec)
```

### Arrow Reversal

To convert an **incoming** edge to an **outgoing** edge, contract with \(g\):

$$\tilde{T}^{m'}_{\ldots} = \sum_m g^{(j)}_{m, m'}\, T^{\ldots}_{m}$$

To convert an **outgoing** edge to an **incoming** edge, contract with \(g^{-1}\):

$$\tilde{T}_{\ldots m'} = \sum_m \left(g^{(j)}\right)^{-1}_{m', m}\, T^{\ldots m}$$

The two operations differ by the Frobenius-Schur phase \((-1)^{2j}\) due to the
identity \(g^{-1} = (-1)^{2j} g^\top\).

## Frobenius-Schur Indicator

The **Frobenius-Schur indicator** of SU(2) representations is:

$$\text{FS}(j) = (-1)^{2j} = \begin{cases} +1 & j \in \mathbb{Z} \\ -1 & j \in \mathbb{Z} + \tfrac{1}{2} \end{cases}$$

This indicator appears as a phase factor whenever a contracted edge pair has its arrows
flipped. In the X-symbol bond inversion scheme, flipping a contracted pair with spin
\(j\) multiplies the X-symbol by \((-1)^{2j}\).

## See Also

- [Clebsch-Gordan Coefficients](cg.md): CG phases are the building blocks of the metric
- [Canonical Basis](../symbols/canonical-basis.md): Conjugate basis computed via
  `compute_conjugate`
- [Direction](../core/direction.md): The arrow direction that the metric transforms
- [Yuzuha Protocol — Arrow Conventions](../../protocol/conventions/arrows.md): Full
  specification of the metric convention

## Notes

The metric \(g^{(j)}\) is **not** the standard Euclidean metric. It is the unique
(up to a global phase) SU(2)-invariant bilinear form on \(V_j\). Using the ordinary
identity in place of \(g\) gives incorrect arrow-reversal results.
