# Yuzuha Protocol

The **Yuzuha Protocol** is a formal specification for recoupling-coefficient libraries
in tensor network applications. It defines the abstract types, self-consistency
requirements, and function interfaces that any conforming implementation must satisfy.

## Motivation

Yuzuha currently implements recoupling theory for the **SU(2)** symmetry group.
However, the interface it provides is intentionally designed to generalize to other
compact Lie groups, particularly SU(N) for \(N > 2\). The Yuzuha Protocol makes this
generalization explicit by separating the abstract interface from the concrete SU(2)
implementation.

A library that conforms to the Yuzuha Protocol:

1. Provides the required abstract types ([Type System](types.md))
2. Implements the required symbol functions ([Symbol Interface](symbols.md))
3. Implements the required duality functions ([Duality Interface](duality.md))
4. Adopts a self-consistent internal convention — the [SU(2) Conventions](conventions/index.md) section documents one valid choice used by the reference implementation
5. Satisfies the caching requirements if caching is implemented ([Data Caching](caching.md))

The current `yuzuha` Python package is the **reference implementation** of this protocol
for SU(2).

## Scope

The protocol covers:

- The **representation label** type — how irreducible representations are identified
- The **edge** type — how labeled, directed edges are represented
- The **specification** type — how a full fusion tree is described
- The **contraction** type — how edge-pair contractions are specified
- The **canonical basis** function — normalization and fusion-tree conventions
- The **X-symbol** function — its required mathematical properties and API shape
- The **R-symbol** function — its required unitarity and composition properties
- The **FS phase** function — its required properties and relationship to arrow reversal
- The **conjugate spec** function — its required properties and relationship to duality
- The **caching layer** — determinism, thread-safety, and key-space requirements

The protocol does **not** prescribe:

- Implementation language or performance characteristics
- Internal algorithms (Racah formula, Wigner 3j/6j, etc.)
- Specific phase choices, fusion-tree coupling order, or OM-index labeling conventions —
  any self-consistent choice is valid; the [SU(2) Conventions](conventions/index.md) section
  documents the choices made by the reference implementation
- The file format of any cache
- How floating-point precision is handled beyond the requirement that all outputs are
  real-valued

## Conformance

A library **conforms** to the Yuzuha Protocol if:

1. All required types are present and satisfy the interface constraints in
   [Type System](types.md)
2. All outputs satisfy the mathematical identities specified in
   [Symbol Interface](symbols.md) and [Duality Interface](duality.md)
3. All error conditions specified in [Symbol Interface](symbols.md) and
   [Duality Interface](duality.md) are raised with appropriate exceptions
4. If caching is implemented, it satisfies [Data Caching](caching.md)

Conformance may be verified using the protocol's reference test suite (to be published
separately).

## Terminology

The following terms are used throughout this document with their precise meanings:

| Term | Definition |
|------|-----------|
| **Representation label** | An element of the set of irreducible representations of the symmetry group |
| **Edge** | A pair (representation label, direction) identifying one external edge of a tensor |
| **Spec** | A complete description of a CG tensor: edges and internal representations |
| **Internal representation** | An intermediate representation appearing in a fusion tree |
| **OM dimension** | Number of valid internal representation paths for a given spec |
| **OM index** | Integer index \(\mu\) labeling one such path |
| **Canonical basis** | The chosen orthonormal basis for the space of CG tensors with a given spec |
| **X-symbol** | Recoupling coefficient tensor for contracting two CG tensors |
| **R-symbol** | Transformation matrix for permuting the edges of a CG tensor |

## Document Structure

| Page | Contents |
|------|---------|
| [Type System](types.md) | Required types: `RepLabel`, `Direction`, `Edge`, `Spec`, `Contraction` |
| [Symbol Interface](symbols.md) | `canonical_basis`, `compute_x_symbol`, `compute_r_symbol` — signatures and properties |
| [Duality Interface](duality.md) | `fs_phase`, `compute_conjugate` — Frobenius-Schur phase and spec dualisation |
| [SU(2) Conventions](conventions/index.md) | One valid self-consistent convention: phase choices, fusion-tree form, metric |
| [Data Caching](caching.md) | Key spaces, determinism, thread-safety |
