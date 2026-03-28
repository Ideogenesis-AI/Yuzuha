# Yuzuha: SU(2) Recoupling Engine

Welcome to the documentation for **Yuzuha**, a high-performance library for computing
SU(2) recoupling coefficients in quantum many-body physics and tensor network applications.

## What is Yuzuha?

Yuzuha provides efficient computation of X-symbols and R-symbols — the fundamental building
blocks for contracting and permuting symmetry-constrained tensors in SU(2)-symmetric tensor
networks. Created as part of the Ideogenesis-AI effort in studying quantum many-body systems,
Yuzuha is optimized as the recoupling engine for the [Nicole](https://github.com/Ideogenesis-AI/Nicole)
tensor library while remaining fully usable as a standalone library.

The library is implemented in **Rust** for performance and exposed to **Python** via
[PyO3](https://pyo3.rs/), providing a zero-copy NumPy interface. Computed canonical CG bases
are cached in a local SQLite database, making repeated calculations near-instantaneous.

## Key Features

- **X-symbols**: Recoupling coefficients for arbitrary tensor network contractions, mapping
  outer-multiplicity (OM) indices of two input CG tensors to the OM index of their fused output
- **R-symbols**: Axis permutation transformations on CG tensors, represented as unitary matrices
  on the OM space
- **Canonical CG bases**: Left-associative fusion-tree bases for arbitrary spin configurations,
  with systematic OM enumeration
- **CG coefficient engine**: Real-valued Condon-Shortley CG coefficients with automatic
  in-memory caching
- **Invariant metric**: Full support for arbitrary arrow directions via the \((-1)^{j-m}\)
  metric tensor
- **SQLite caching**: Persistent, thread-safe database for canonical bases, X-symbols,
  and R-symbols
- **NumPy integration**: Symbols returned as `float64` NumPy arrays, compatible with
  standard scientific Python workflows
- **Rust core**: High-performance Rust implementation with a clean Python API

## Relationship to the Yuzuha Protocol

Yuzuha currently implements the **SU(2)** case of recoupling theory. The interface it
provides is designed to generalize to other groups SU(N). The
[Yuzuha Protocol](../protocol/index.md) formally specifies this abstract interface — it
defines the types, conventions, and function signatures that any conforming SU(N)
recoupling library must satisfy, with the present SU(2) implementation as the reference.

## Getting Started

New to Yuzuha? Follow these steps:

1. **[Installation](installation.md)**: Install Yuzuha via pip or build from source with Maturin
2. **[Core Concepts](core-concepts.md)**: Understand spins, edges, CG specifications, and the
   OM index
3. **[API Reference](../api/index.md)**: Detailed documentation for all types and functions

## Use Cases

Yuzuha is designed for researchers and developers working on:

- **Tensor Network Algorithms**: DMRG, TEBD, PEPS, and other symmetry-aware tensor methods
- **Quantum Many-Body Physics**: Systems with SU(2) spin symmetry (e.g. spin-\(\frac{1}{2}\)
  Heisenberg models, spinful fermion systems like the Hubbard models)
- **Condensed Matter Theory**: Frustrated spin systems or strongly correlated spinful fermionic systems with angular momentum conservation
- **Quantum Chemistry**: Multi-reference methods with spin-adapted bases

## License

Yuzuha is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are
free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We
encourage you to share any improvements you make back to the community, helping Yuzuha grow and
benefit all users. See the
[LICENSE](https://github.com/Ideogenesis-AI/Yuzuha/blob/stable/LICENSE) file for the full
license text. For more information about GPL-3.0, visit
<https://www.gnu.org/licenses/gpl-3.0.html>
