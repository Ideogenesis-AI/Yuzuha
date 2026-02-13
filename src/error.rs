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

//! Error types for the Yuzuha library

use thiserror::Error;

/// Errors that can occur during SU(2) X-symbol computations
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum YuzuhaError {
    /// Invalid spin quantum number (must be non-negative)
    #[error("Invalid spin quantum number: {0}")]
    InvalidSpin(i32),

    /// Triangle inequality violated for three angular momenta
    #[error("Triangle inequality violated: j1={0}, j2={1}, j3={2}")]
    TriangleViolation(i32, i32, i32),

    /// Magnetic quantum number out of valid range [-J, +J]
    #[error("Magnetic number {m} out of range for spin J={j}")]
    InvalidMagneticNumber { j: i32, m: i32 },

    /// Contraction specification is invalid
    #[error("Invalid contraction: {0}")]
    InvalidContraction(String),

    /// Leg ID not found in CGT specification
    #[error("Leg ID not found: {0}")]
    LegNotFound(String),

    /// Incompatible leg spins for contraction
    #[error("Incompatible leg spins for contraction: J1={0}, J2={1}")]
    IncompatibleLegs(i32, i32),

    /// Invalid CGT specification
    #[error("Invalid CGT specification: {0}")]
    InvalidCGTSpec(String),

    /// Tensor dimension mismatch
    #[error("Tensor dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Edge index out of bounds
    #[error("Edge index {0} out of bounds (number of edges: {1})")]
    IndexOutOfBounds(usize, usize),
}

/// Result type for Yuzuha operations
pub type Result<T> = std::result::Result<T, YuzuhaError>;
