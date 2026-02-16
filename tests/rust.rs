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

// Integration tests are organized in the rust/ subdirectory
// This file ensures Cargo can discover them

#[path = "rust/test_cgbasis.rs"]
mod test_cgbasis;

#[path = "rust/test_cgspec.rs"]
mod test_cgspec;

#[path = "rust/test_database.rs"]
mod test_database;

#[path = "rust/test_fsymbol.rs"]
mod test_fsymbol;

#[path = "rust/test_primitives.rs"]
mod test_primitives;

#[path = "rust/test_racah.rs"]
mod test_racah;

#[path = "rust/test_xsymbol.rs"]
mod test_xsymbol;
