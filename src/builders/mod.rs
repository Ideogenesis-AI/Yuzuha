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

//! Builder functions for SU(2) tensor networks
//!
//! Provides atomic builders (CG3, connectors), OM enumeration, canonical basis
//! construction, and X-symbol computation.

pub mod atomic;
pub mod builders;
pub mod cache;
pub mod dualize;
pub mod om_basis;
pub mod rsymbol;
pub mod xsymbol;

// API exports
pub use atomic::{build_cg3, build_connector};
pub use builders::build_canonical_basis_data;
pub use om_basis::{enumerate_alpha, om_dimension};
pub use rsymbol::{compute_rsymbol, RSymbol};
pub use xsymbol::{compute_xsymbol, XSymbol};

// Only available under cfg(test) or with the `test-utils` feature.
// Integration tests enable this via: cargo test --features test-utils
#[cfg(any(test, feature = "test-utils"))]
pub use cache::TestCacheGuard;
