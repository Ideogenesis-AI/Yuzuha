# SU(2) Conventions

This section documents the mathematical conventions used by the **SU(2) reference
implementation** of the Yuzuha Protocol. The protocol requires each implementation to
adopt a self-consistent convention, but does not mandate any specific choice. The
conventions below represent one valid option; alternative implementations are free to
use different phase choices, fusion-tree coupling orders, or OM-index labelings provided
they remain internally consistent and their APIs satisfy the
[Symbol Interface](../symbols.md) and [Duality Interface](../duality.md).

## Structure

The conventions are divided into three pages, each covering a logically distinct aspect
of the SU(2) implementation:

| Page | Contents |
|------|---------|
| [Fusion Conventions](fusion.md) | Left-associative fusion tree; lexicographic OM-index ordering |
| [Basis Conventions](basis.md) | Condon-Shortley CG coefficients; orthonormal basis normalization; canonical edge directions |
| [Arrow Conventions](arrows.md) | Charge-conservation sign rule; invariant metric; arrow reversal; Frobenius-Schur indicator; canonical conjugate pattern |

## Fusion Conventions

The SU(2) reference implementation uses a **left-associative** fusion tree. For an
\(n\)th-order tensor with external representations \((j_1, \ldots, j_n)\), the
internal representation sequence \((\alpha_1, \ldots, \alpha_{n-2})\) is built
iteratively from the left:

$$\bigl(\cdots\bigl((j_1 \otimes j_2) \otimes j_3\bigr) \otimes \cdots\bigr) \otimes j_n$$

Valid internal sequences are enumerated in **lexicographic order**, with the OM index
\(\mu = 0\) assigned to the smallest sequence. See [Fusion Conventions](fusion.md) for
the full specification.

## Basis Conventions

Clebsch-Gordan coefficients follow the **Condon-Shortley phase convention**:

$$C^{j_3 m_3}_{j_1 m_1,\, j_2 m_2} = \langle j_1 m_1,\, j_2 m_2 \mid j_3 m_3 \rangle$$

Their values are computed using the **Racah formula**, which expresses each CG
coefficient as an explicit algebraic sum over factorials of angular momentum quantum
numbers.

The canonical basis tensors \(\{B_\mu\}\) form an **orthonormal** set over the CG
tensor space. For a 3rd-order tensor, the single basis element is normalized by
\(1/\sqrt{2j_3 + 1}\) relative to the bare CG coefficient.

Each basis tensor is computed in a fixed **canonical direction**: all **leading edges**
(first \(n-1\) edges) are **incoming** and the **terminal edge** (last edge) is
**outgoing**. Bases with other direction combinations are derived by contracting
deviating edges with the invariant metric \(g^{(j)}\). See
[Basis Conventions](basis.md) for the full specification.

## Arrow Conventions

The `Direction` type assigns \(+1\) to `Incoming` and \(-1\) to `Outgoing` at the
protocol level (see [Type System — `Direction`](../types.md#direction)). The signed
magnetic-number sum must therefore vanish for any non-zero basis element:

$$\sum_{\text{incoming}} m_i - \sum_{\text{outgoing}} m_j = 0$$

The **invariant metric** \(g^{(j)}\) mediates between a representation and its dual:

$$g^{(j)}_{m,\,m'} = (-1)^{j-m}\,\delta_{m,\,-m'}$$

It satisfies \((g^{(j)})^2 = (-1)^{2j}\,I\), so its square equals the
**Frobenius-Schur (FS) indicator** \(\text{FS}(j) = (-1)^{2j}\). The FS indicator is
\(+1\) for integer spin and \(-1\) for half-integer spin; it sets the phase acquired
per bond inversion. See [Arrow Conventions](arrows.md) for the full specification.
