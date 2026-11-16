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

[Fusion Conventions](fusion.md) fixes the shape of the multiplicity space. Representations
are coupled left-associatively, and the resulting internal sequences are enumerated
lexicographically so that an OM index remains consistent on every call and in every
cache generation.

[Basis Conventions](basis.md) fixes the numbers. Clebsch-Gordan coefficients follow the
Condon-Shortley phase convention and are evaluated with the Racah formula; the basis
tensors built from them are normalized to be orthonormal; and each is constructed in
one fixed arrow pattern, with all leading edges incoming and the terminal edge outgoing.

[Arrow Conventions](arrows.md) fixes the signs. It gives the invariant metric that relates
a representation to its dual, the rules for raising and lowering an index with it, and the
Frobenius-Schur indicator \(\text{FS}(j) = (-1)^{2j}\) that is picked up per inverted bond
and per deviating edge in a conjugation.
