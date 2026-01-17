# Yuzuha: an SU(2) Clebsch–Gordan Engine

Yuzuha implements a from-scratch algorithm for computing SU(2) X-symbols (fusion category morphisms) used in tensor network computations, particularly in symmetric tensor network simulations for quantum lattice models.

## Features

- **Arbitrary Contraction Patterns**: Supports any subset of legs with any arrow directions
- **Clebsch-Gordan Coefficients**: Uses Condon-Shortley convention (real-valued)
- **Arrow Reversal**: Handles direction changes via the invariant metric g^(j)
- **Left-Associative Fusion Trees**: Systematic tree construction for combining spins
- **QSpace Sign Convention**: Unique basis CGT fixing for deterministic results
- **Direct Tensor Network Computation**: Computes X-symbols as small tensor-network contractions in m-space

## Algorithm Design

The implementation follows these key design principles:

### 1. Spin Representation
- Uses doubled integers to avoid half-floats (e.g., j=1/2 → J=1)
- Magnetic quantum numbers M ∈ {-J, -J+2, ..., J}

### 2. Building Blocks
- **CG Coefficients**: ⟨j₁ m₁, j₂ m₂ | j₃ m₃⟩ with selection rules
- **Invariant Metric**: g^(J)_{M,M'} = (-1)^((J-M)/2) δ_{M,-M'}
- **OM Basis Enumeration**: Outer multiplicity indices via internal spin tuples

### 3. CGT (Coupled Group Theory) Basis Elements
- Left-associative fusion tree structure
- Sorted sign convention for uniqueness
- Efficient amplitude computation without building full tensors

### 4. X-Symbol Computation
The core computation:
```
X^γ_αβ = ⟨C^(γ) | (A^(α) contract B^(β))⟩
```
where α, β, γ are outer multiplicity (OM) indices corresponding to internal spin tuples.

### Implementation Strategy
- Build small tensor networks with trivalent CG vertices
- Insert connector tensors (identity or g-delta) for contractions
- Use generic tensor contraction (greedy pairwise tensordot)
- Support arbitrary contraction patterns automatically

## Status

🚧 **Under Development** 🚧

This library is currently in the early planning/implementation phase. The algorithm specification is complete (see `algthm/x-symbol.md`), and implementation is in progress.

## Technical Background

X-symbols are fundamental objects in the representation theory of fusion categories. In the context of SU(2):

- They describe how to decompose composite tensor network operations
- Essential for efficient tensor network algorithms (PEPS, MERA, etc.)
- Provide the "recoupling theory" needed for arbitrary contractions
- Real-valued for SU(2) (unlike general quantum groups)

## Algorithm Features

### Correctness Guarantees
- **Identity contraction test**: ⟨C|A⟩ yields Kronecker delta in OM space
- **Arrow flip consistency**: Double arrow flip preserves X
- **Reality**: All X-symbols are real (within floating-point precision)

### Efficiency
- No full tensor materialization for CGT amplitudes
- Small tensor dimensions (SU(2) dims are 2j+1)
- Direct network contraction without symbolic recoupling
- Works for any contraction pattern without case-by-case rewriting

## License

Yuzuha is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license.

See the [LICENSE](LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html
