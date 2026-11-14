// Copyright (C) 2026 Changkai Zhang.
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

//! Shared fixtures for the integration test suite.
//!
//! Helpers live here when more than one `tests/rust/*.rs` module needs them.
//! Module-specific scaffolding stays in the module that uses it.

// Each integration module uses a different subset of these.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use approx::assert_relative_eq;
use ndarray::{ArrayD, Axis};
use yuzuha::builders::builders::build_canonical_basis_data;
use yuzuha::builders::TestCacheGuard;
use yuzuha::core::{CGSpec, Edge, MagneticNumber, Spin};

/// Serializes the tests that redirect the cache through `YUZUHA_CACHE_PATH`.
static CACHE_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Point the cache at a private temporary directory for every thread.
///
/// `TestCacheGuard` is the right tool for a single-threaded test, but its
/// override is thread-local, so a test that spawns workers and expects them to
/// share one database has to set the environment variable instead. That
/// variable is process-global, and Cargo runs the integration tests as threads
/// of one process: while it is set, any test that resolves a cache path
/// without a `TestCacheGuard` is silently redirected too. Holding this guard
/// serializes such tests against each other, and restores both the previous
/// value and the temporary directory on drop.
pub struct CacheEnvGuard {
    _lock: MutexGuard<'static, ()>,
    previous: Option<String>,
    dir: PathBuf,
}

impl CacheEnvGuard {
    /// Acquire the lock and redirect the cache into a fresh directory.
    pub fn new(prefix: &str) -> Self {
        // A test that panicked while holding the lock poisons it, but it also
        // restored the variable on unwind, so the state is still usable.
        let lock = CACHE_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        // A plain timestamp is not unique enough on a microsecond clock, so a
        // counter distinguishes directories created in the same instant.
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let dir = std::env::temp_dir().join(format!(
            "{prefix}_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("Failed to create temp dir");

        let previous = std::env::var("YUZUHA_CACHE_PATH").ok();
        unsafe {
            std::env::set_var("YUZUHA_CACHE_PATH", &dir);
        }

        CacheEnvGuard { _lock: lock, previous, dir }
    }

    /// Directory the cache is redirected to.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for CacheEnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var("YUZUHA_CACHE_PATH", value),
                None => std::env::remove_var("YUZUHA_CACHE_PATH"),
            }
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Build a spin from its doubled value, panicking on invalid input.
///
/// `j(1)` is j=1/2, `j(2)` is j=1, and so on.
pub fn j(doubled: i32) -> Spin {
    Spin::new(doubled).expect("valid doubled spin")
}

/// Build a `CGSpec` from doubled spin values, with the last edge outgoing.
pub fn create_spec(doubled_values: Vec<i32>) -> CGSpec {
    let last = doubled_values.len() - 1;
    let edges: Vec<Edge> = doubled_values
        .iter()
        .enumerate()
        .map(|(i, &doubled)| {
            let spin = j(doubled);
            if i == last {
                Edge::outgoing(spin)
            } else {
                Edge::incoming(spin)
            }
        })
        .collect();
    CGSpec::from_edges(edges).expect("edge spins must couple to j=0")
}

/// Render an alpha (internal spin configuration) as `(j1, j2, ...)`.
pub fn format_alpha(alpha: &[Spin]) -> String {
    let parts: Vec<String> = alpha.iter().map(|s| s.to_string()).collect();
    format!("({})", parts.join(", "))
}

/// Assert that the canonical basis for `edges` is orthonormal.
///
/// Checks two properties of `build_canonical_basis_data`, for any number of
/// edges and any mix of directions:
///
/// 1. Every OM slice has unit Frobenius norm.
/// 2. Contracting two OM slices over all but the terminal edge gives
///    `delta_{alpha,beta} / (2j+1) * I`, where j is the terminal edge spin.
///    The `1/(2j+1)` is what the `sqrt(2j+1)` normalization leaves on the
///    diagonal.
pub fn assert_canonical_basis_orthonormal(edges: Vec<Edge>) {
    let _guard = TestCacheGuard::new();

    let n = edges.len();
    assert!(n >= 2, "canonical basis needs at least two edges");
    let dim_out = edges[n - 1].dimension();

    let external_dims: Vec<usize> = edges.iter().map(|e| e.dimension()).collect();

    let spec = CGSpec::from_edges(edges).expect("edge spins must couple to j=0");
    let data = build_canonical_basis_data(&spec).expect("basis construction must succeed");
    let om_dim = spec.om_dimension();

    let mut expected_shape = external_dims;
    expected_shape.push(om_dim);
    assert_eq!(data.shape(), expected_shape.as_slice());

    // The OM axis is appended after the n external axes.
    let om_axis = Axis(n);
    let contracted: Vec<Axis> = (0..n - 1).map(Axis).collect();
    let expected_diag = 1.0 / (dim_out as f64);

    for alpha in 0..om_dim {
        let slice = data.index_axis(om_axis, alpha);
        let frobenius_sq: f64 = slice.mapv(|x| x * x).sum();
        assert_relative_eq!(frobenius_sq.sqrt(), 1.0, epsilon = 1e-10);
    }

    for alpha in 0..om_dim {
        for beta in 0..om_dim {
            let slice_alpha = data.index_axis(om_axis, alpha);
            let slice_beta = data.index_axis(om_axis, beta);

            let result =
                ndarray_einsum::tensordot(&slice_alpha, &slice_beta, &contracted, &contracted);

            for i in 0..dim_out {
                for k in 0..dim_out {
                    let expected = if alpha == beta && i == k { expected_diag } else { 0.0 };
                    assert_relative_eq!(result[[i, k]], expected, epsilon = 1e-10);
                }
            }
        }
    }
}

/// Collapse a canonical basis into a single tensor using per-OM weights.
///
/// Produces `result[...] = sum_alpha weights[alpha] * basis[..., alpha]`, the
/// generic way to build a test tensor that is not aligned with any single OM
/// configuration.
pub fn build_weighted_tensor(basis: &ArrayD<f64>, weights: &[f64]) -> ArrayD<f64> {
    let om_axis = Axis(basis.ndim() - 1);
    assert_eq!(basis.shape()[om_axis.index()], weights.len(), "one weight per OM index");

    let mut result_shape = basis.shape().to_vec();
    result_shape.pop();
    let mut result = ArrayD::zeros(result_shape);

    for (alpha, &weight) in weights.iter().enumerate() {
        let slice = basis.index_axis(om_axis, alpha);
        result = result + &(slice.to_owned() * weight);
    }
    result
}

/// Reverse one tensor axis through the invariant metric.
///
/// Mirrors the axis (m goes to -m) and applies the direction-reversal phase:
/// `(-1)^{j+m}` for incoming to outgoing, with an extra `(-1)^{2j}` going the
/// other way.
pub fn invert_edge_direction_axis(
    tensor: &ArrayD<f64>,
    axis_idx: usize,
    spin: Spin,
    from_incoming: bool,
) -> ArrayD<f64> {
    let mut result = tensor.clone();
    result.invert_axis(Axis(axis_idx));

    for m_idx in 0..spin.dimension() {
        let m = MagneticNumber::from_index(m_idx, spin);
        let phase_jm = if (spin.twice() + m.twice()) % 4 == 0 { 1.0 } else { -1.0 };
        let phase = if from_incoming {
            phase_jm
        } else {
            let phase_2j = if spin.twice() % 2 == 0 { 1.0 } else { -1.0 };
            phase_2j * phase_jm
        };
        result.index_axis_mut(Axis(axis_idx), m_idx).mapv_inplace(|x| x * phase);
    }

    result
}
