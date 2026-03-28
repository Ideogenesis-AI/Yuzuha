# Basis Conventions

## Clebsch-Gordan Coefficients

The SU(2) reference implementation uses the **Condon-Shortley convention** for
Clebsch-Gordan (CG) coefficients:

$$C^{j_3 m_3}_{j_1 m_1,\, j_2 m_2} = \langle j_1 m_1,\, j_2 m_2 \mid j_3 m_3 \rangle$$

Their values are computed using the **Racah formula**, which expresses each CG
coefficient as an explicit algebraic sum over factorials of angular momentum quantum
numbers.

Required properties:

1. **Real-valued**: All CG coefficients are real numbers.
2. **Orthonormality**: \(\displaystyle\sum_{m_1, m_2} C^{j_3 m_3}_{j_1 m_1, j_2 m_2}\, C^{j_3' m_3'}_{j_1 m_1, j_2 m_2} = \delta_{j_3 j_3'}\,\delta_{m_3 m_3'}\)
3. **Phase choice**: The Condon-Shortley phase convention fixes the relative signs
   of CG coefficients so that the \(m_1 = j_1\), \(m_2 = j_2 - j_3 + j_1\)
   coefficient (highest available) is positive.

For SU(N) with \(N > 2\), the analogous reduced matrix elements (Wigner coefficients)
must be defined with an analogous real, orthonormal, and positivity-compatible phase
convention. The precise generalization is group-specific and must be documented in the
conforming implementation.

## Canonical Basis Normalization

The canonical basis tensor \(B_\mu\) for OM index \(\mu\) must be normalized so that
the set \(\{B_\mu\}\) forms an **orthonormal** basis of the CG tensor space:

$$\langle B_\mu | B_\nu \rangle \equiv \sum_{m_1, \ldots, m_n} B_\mu(m_1, \ldots, m_n)\, B_\nu(m_1, \ldots, m_n) = \delta_{\mu\nu}$$

**3rd-order normalization**: For a 3rd-order tensor (\(n = 3\)), the OM dimension is always
1. The single basis element \(B_0\) acquires a normalization factor relative to the
bare CG coefficient \(C^{j_3 m_3}_{j_1 m_1, j_2 m_2}\):

$$B_0(m_1, m_2, m_3) = \frac{1}{\sqrt{2j_3 + 1}}\, C^{j_3 m_3}_{j_1 m_1, j_2 m_2}$$

This factor ensures that the squared norm over all magnetic indices is 1.

## Canonical Basis Direction

For an \(n\)th-order tensor we define:

- **Leading edges**: the first \(n - 1\) external edges
- **Terminal edge**: the last external edge

The SU(2) reference implementation builds each basis tensor from a fixed
**canonical direction**: all **leading edges incoming** and the **terminal edge
outgoing**. This is the reference configuration for which the fusion-tree CG
contraction is performed directly.

When `canonical_basis` is called with a spec whose edge directions deviate from this
canonical pattern, the implementation derives the result from the canonical basis by
contracting each deviating edge \(k\) (with spin \(j_k\)) with the invariant metric
\(g^{(j_k)}\) (see [Arrow Conventions — Invariant Metric](arrows.md#invariant-metric)):

$$B^S_\mu = \left(\prod_{k \,\in\, \text{deviating}(S)} g^{(j_k)}\right) \cdot B^{\text{canonical}}_\mu$$

Each metric contraction on a deviating edge contributes a factor of
\(\text{FS}(j_k) = (-1)^{2j_k}\) when the direction is subsequently inverted (bond
inversion). The returned tensor already incorporates these metric contractions, so
callers receive the correct basis for the given spec regardless of direction.
