// Copyright (C) 2025-2026 Changkai Zhang.
//
// This file is part of Yuzuha library.
//
// Yuzuha is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
//
// Yuzuha is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Yuzuha. If not, see <https://www.gnu.org/licenses/>.

//! # Yuzuha: SU(2) X-symbols for Tensor Networks
//!
//! Yuzuha is a library for computing SU(2) X-symbols (also known as recoupling coefficients)
//! for arbitrary tensor network contractions using left-associative fusion trees.
//!
//! ## Features
//!
//! - **Clebsch-Gordan coefficients**: Real-valued Condon-Shortley convention with caching
//! - **Invariant metric**: Arrow reversal support for arbitrary leg directions
//! - **Outer multiplicity enumeration**: Systematic enumeration of internal spin configurations
//! - **CGT amplitudes**: Efficient computation without materializing full tensors
//! - **Tensor networks**: Generic contraction with greedy optimization
//! - **X-symbols**: Full computation for arbitrary contraction patterns
//!
//! ## Quick Start
//!
//! ```rust
//! use yuzuha::{Spin, Edge, CGSpec, Contraction, compute_xsymbol};
//!
//! // Create two CG specifications with spin edges
//! let j_half = Spin::new(1).unwrap(); // j=1/2 (doubled: J=1)
//! let j1 = Spin::new(2).unwrap();     // j=1 (doubled: J=2)
//!
//! // Automatically enumerates all OM configurations
//! // Uses deterministic Condon-Shortley convention
//! // Requires at least 3 edges for canonical basis computation
//! let spec_a = CGSpec::from_edges(vec![
//!     Edge::incoming(j_half),
//!     Edge::incoming(j_half),
//!     Edge::incoming(j1),
//! ]).unwrap();
//!
//! let spec_b = CGSpec::from_edges(vec![
//!     Edge::incoming(j1),
//!     Edge::incoming(j_half),
//!     Edge::incoming(j_half),
//! ]).unwrap();
//!
//! // Contract edge 2 from spec_a (j=1) with edge 0 from spec_b (j=1)
//! // Similar to: numpy.tensordot(A, B, axes=([2], [0]))
//! let contraction = Contraction::new(&[2], &[0]);
//!
//! // Compute X-symbol
//! let x = compute_xsymbol(&spec_a, &spec_b, &contraction).unwrap();
//!
//! println!("X-symbol dimensions: {:?}", x.dimensions());
//! ```
//!
//! ## Conventions
//!
//! - **Spin quantum numbers**: Stored as doubled integers (j=1/2 → J=1, j=1 → J=2)
//! - **Magnetic numbers**: Also doubled integers, M ∈ {-J, -J+2, ..., J}
//! - **Arrow directions**: Incoming (+1) and Outgoing (-1)
//! - **Fusion trees**: Left-associative by default
//! - **Sign convention**: QSpace sorted-sign for uniqueness
//!
//! ## Modules
//!
//! - [`core`]: Core data structures (Spin, Edge, CGSpec, CGTensor, etc.)
//! - [`primitives`]: CG coefficients, metric, triangle rules
//! - [`cgt_amplitude`]: CGT amplitude computation
//! - [`builders`]: OM enumeration, atomic builders (CG3, connectors), canonical basis construction, and X-symbol computation

pub mod builders;
pub mod core;
pub mod error;
pub mod primitives;

// Re-export commonly used types
pub use builders::{
    build_canonical_basis_data, build_cg3, compute_rsymbol, compute_xsymbol, enumerate_alpha,
    om_dimension, RSymbol, XSymbol,
};
pub use core::{CGSpec, CGTensor, Contraction, Direction, Edge, MagneticNumber, Spin};
pub use error::{Result, YuzuhaError};
