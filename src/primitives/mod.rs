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

//! Primitive operations for SU(2) representation theory
//!
//! This module provides the building blocks for computing X-symbols:
//! - Triangle rules for angular momentum coupling
//! - Invariant metric (arrow reversal)
//! - Clebsch-Gordan coefficients

pub mod cg;
pub mod metric;
pub mod triangle;

pub use cg::clebsch_gordan;
pub use metric::{g, use_as_incoming, use_as_outgoing};
pub use triangle::{allowed_triangle, triangle_check};
