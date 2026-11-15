# Clebsch-Gordan Coefficients

Clebsch-Gordan coefficients are the lowest layer Yuzuha builds on: every canonical basis
is a contraction of them along a fusion tree, and every symbol is a projection onto such
bases. They are computed in the Rust layer and are not exposed as a Python function, so
this page documents the convention and properties.

## Description

A CG coefficient gives the decomposition of the tensor product of two SU(2)
representations into irreducible components:

$$C^{j_3 m_3}_{j_1 m_1,\, j_2 m_2} \equiv \langle j_1 m_1,\, j_2 m_2 \mid j_3 m_3 \rangle$$

Yuzuha uses the **Condon-Shortley convention**, which fixes the relative phases so that
every coefficient is real-valued. That choice allows all of Yuzuha's outputs to be
`float64` arrays without complex involvement.

### Selection Rules

A coefficient whose indices violate any of the following is identically zero, so these
rules also determine which internal spin paths a `CGSpec` admits at all:

| Rule | Condition |
|------|-----------|
| Magnetic conservation | \(m_1 + m_2 = m_3\) |
| Triangle inequality | \(\lvert j_1 - j_2 \rvert \leq j_3 \leq j_1 + j_2\) |
| Integer sum | \(j_1 + j_2 + j_3 \in \mathbb{Z}\) |
| Range | \(-j_i \leq m_i \leq j_i\) for each \(i\) |

### Orthonormality Relations

The coefficients form an orthonormal basis in both directions. Summing over the
uncoupled magnetic numbers gives **column** orthonormality:

$$\sum_{m_1, m_2} C^{j_3 m_3}_{j_1 m_1, j_2 m_2}\, C^{j_3' m_3'}_{j_1 m_1, j_2 m_2}
= \delta_{j_3 j_3'}\, \delta_{m_3 m_3'}$$

Summing over the coupled ones gives **row** orthonormality:

$$\sum_{j_3, m_3} C^{j_3 m_3}_{j_1 m_1, j_2 m_2}\, C^{j_3 m_3}_{j_1 m_1', j_2 m_2'}
= \delta_{m_1 m_1'}\, \delta_{m_2 m_2'}$$

The first relation is what propagates to the orthonormality of the
[canonical basis](../symbols/canonical-basis.md) once the coefficients are contracted
along a fusion tree.

### Internal Access

The Rust layer provides `clebsch_gordan(j1, m1, j2, m2, j3, m3)`, called by
`canonical_basis` as it walks the fusion tree. There is no Python binding: individual
coefficients are an implementation detail, and exposing them would invite callers to
rebuild bases by hand under a different normalization.

### Caching

Coefficients are cached in-process by the Rust layer (`once_cell`), keyed on the full
index tuple \((j_1, m_1, j_2, m_2, j_3, m_3)\). Individual values are never written to
disk; what persists on disk is the assembled canonical basis tensor, which is the unit of
reuse that matters in practice.

## See Also

- [Canonical Basis](../symbols/canonical-basis.md): Uses CG coefficients to build the
  full CG tensor for a `CGSpec`
- [Metric Tensor](metric.md): Arrow-reversal transformation built on CG phases
- [Triangle Rules](triangle.md): Triangle-inequality utilities
- [Yuzuha Protocol — Basis Conventions](../../protocol/conventions/basis.md): Phase
  convention requirements

## Notes

The implementation evaluates the standard Racah formula, an explicit algebraic sum over
factorials of the angular momentum quantum numbers, and caches the result. Large spins
are not a performance concern in practice: the cost is paid once per index tuple and the
cached canonical bases absorb the rest.
