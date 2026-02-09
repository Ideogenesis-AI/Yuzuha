# Yuzuha: SU(2) X-symbols for Tensor Networks

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Yuzuha is a Rust library for computing SU(2) X-symbols (also known as recoupling coefficients or 6j-symbols) for arbitrary tensor network contractions using left-associative fusion trees.

## Features

- ✅ **Clebsch-Gordan coefficients**: Real-valued Condon-Shortley convention with automatic caching
- ✅ **Invariant metric**: Full support for arbitrary arrow directions via metric tensor
- ✅ **Outer multiplicity (OM) enumeration**: Systematic enumeration of internal spin configurations
- ✅ **CGT amplitudes**: Efficient computation without materializing full tensors
- ✅ **Tensor networks**: Generic contraction engine with greedy optimization
- ✅ **X-symbols**: Complete computation for arbitrary contraction patterns
- ✅ **Type safety**: Extensive use of Rust's type system for correctness
- ✅ **Comprehensive tests**: >95% code coverage with unit and integration tests

## Quick Start

Add Yuzuha to your `Cargo.toml`:

```toml
[dependencies]
yuzuha = "0.1"
```
## Conventions

### Clebsch-Gordan Coefficients

Yuzuha uses the **Condon-Shortley convention** for CG coefficients:

⟨j₁ m₁, j₂ m₂ | j₃ m₃⟩

Properties:
- Real-valued
- Orthonormal: Σ_{m₁,m₂} CG² = 1
- Selection rules: m₁ + m₂ = m₃ and triangle inequality

### Arrow Directions

The invariant metric g implements arrow reversal:

g^(j)_{m,m'} = (-1)^(j-m) δ_{m,-m'}

This allows converting between incoming and outgoing legs.

### Fusion Trees

- **Left-associative**: (((J₁ ⊗ J₂) ⊗ J₃) ⊗ ...)
- **Internal spins (α)**: Intermediate coupling results
- **OM indices**: Label different internal spin configurations

## Performance

- **CG coefficient caching**: Automatic memoization for repeated calculations
- **Greedy contraction**: Optimized tensor network contraction order
- **Type-level optimization**: Zero-cost abstractions via Rust's type system

## License

Yuzuha is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping Yuzuha grow and benefit all users. See the [LICENSE](LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html