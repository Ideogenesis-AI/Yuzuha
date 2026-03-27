# Clebsch-Gordan Coefficients

Real-valued SU(2) Clebsch-Gordan coefficients in the Condon-Shortley convention.

## Description

Clebsch-Gordan (CG) coefficients describe the decomposition of the tensor product of
two SU(2) representations into irreducible components:

$$C^{j_3 m_3}_{j_1 m_1,\, j_2 m_2} \equiv \langle j_1 m_1,\, j_2 m_2 \mid j_3 m_3 \rangle$$

Yuzuha computes CG coefficients using the **Condon-Shortley convention**, in which all
coefficients are real-valued. They are accessed internally by `canonical_basis` and
are cached in-memory for repeated evaluations.

### Condon-Shortley Convention

The Condon-Shortley convention fixes the relative phases of CG coefficients so that:

1. All coefficients are **real numbers**
2. The coefficients form an **orthonormal** basis:

$$\sum_{m_1, m_2} \langle j_1 m_1, j_2 m_2 \mid j_3 m_3 \rangle^2 = 1$$

3. **Selection rules** are enforced automatically:
   - Magnetic number conservation: \(m_1 + m_2 = m_3\)
   - Triangle inequality: \(|j_1 - j_2| \leq j_3 \leq j_1 + j_2\)
   - Integer total spin: \(j_1 + j_2 + j_3 \in \mathbb{Z}\)

### Internal Access

CG coefficients are not directly exposed as a Python function but are used internally
by `canonical_basis`. The Rust layer provides `clebsch_gordan(j1, m1, j2, m2, j3, m3)`
and caches results in a thread-local in-memory store.

### Caching

CG coefficients are cached at two levels:

1. **In-memory** (Rust, `once_cell`): Per-process cache keyed by
   \((j_1, m_1, j_2, m_2, j_3, m_3)\)
2. **SQLite** (Rust): Canonical basis tensors built from CG coefficients are persisted
   to disk; individual CG values are not stored separately

## Selection Rules

| Rule | Condition |
|------|-----------|
| Magnetic conservation | \(m_1 + m_2 = m_3\) |
| Triangle inequality | \(\lvert j_1 - j_2 \rvert \leq j_3 \leq j_1 + j_2\) |
| Integer sum | \(j_1 + j_2 + j_3 \in \mathbb{Z}\) |
| Range | \(-j_i \leq m_i \leq j_i\) for each \(i\) |

Any coefficient with indices violating these rules is identically zero.

## Orthonormality Relations

**Column orthonormality** (sum over coupled states):

$$\sum_{m_1, m_2} C^{j_3 m_3}_{j_1 m_1, j_2 m_2}\, C^{j_3' m_3'}_{j_1 m_1, j_2 m_2}
= \delta_{j_3 j_3'}\, \delta_{m_3 m_3'}$$

**Row orthonormality** (sum over output states):

$$\sum_{j_3, m_3} C^{j_3 m_3}_{j_1 m_1, j_2 m_2}\, C^{j_3 m_3}_{j_1 m_1', j_2 m_2'}
= \delta_{m_1 m_1'}\, \delta_{m_2 m_2'}$$

## See Also

- [Canonical Basis](../symbols/canonical-basis.md): Uses CG coefficients to build the
  full CG tensor for a `CGSpec`
- [Metric Tensor](metric.md): Arrow-reversal transformation built on CG phases
- [Triangle Rules](triangle.md): Triangle-inequality utilities
- [Yuzuha Protocol — SU(2) Conventions](../../protocol/conventions.md): Phase
  convention requirements

## Notes

The Rust implementation uses the standard Racah formula to compute CG coefficients
from scratch, then caches the results. For large spins (e.g. \(j > 10\)), the
computation is still fast because the cache is populated once and reused.
