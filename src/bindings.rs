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

//! Python bindings for Yuzuha library using PyO3
//!
//! This module provides Python bindings for computing X-symbols and R-symbols
//! with automatic database connection management for caching.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray2, PyArray3, PyArrayDyn};

use crate::core::{Spin as RustSpin, Edge as RustEdge, CGSpec as RustCGSpec, 
                   Contraction as RustContraction, Direction};
use crate::builders::{compute_xsymbol as rust_compute_xsymbol, 
                      compute_rsymbol as rust_compute_rsymbol,
                      build_canonical_basis_data as rust_build_canonical_basis_data};
use crate::builders::dualize::{
    fs_phase_for_spin as rust_fs_phase_for_spin,
    compute_conjugate as rust_compute_conjugate,
};
use crate::error::YuzuhaError;

/// Convert Rust YuzuhaError to Python exception
impl From<YuzuhaError> for PyErr {
    fn from(err: YuzuhaError) -> PyErr {
        match err {
            YuzuhaError::InvalidSpin(j) => {
                PyValueError::new_err(format!("Invalid spin value: {}", j))
            }
            YuzuhaError::InvalidMagneticNumber { m, j } => {
                PyValueError::new_err(format!(
                    "Invalid magnetic number {} for spin {}", m, j
                ))
            }
            YuzuhaError::InvalidCGTSpec(msg) => {
                PyValueError::new_err(format!("Invalid CGSpec: {}", msg))
            }
            YuzuhaError::TriangleViolation(j1, j2, j3) => {
                PyValueError::new_err(format!(
                    "Triangle inequality violation: j1={}, j2={}, j3={}", j1, j2, j3
                ))
            }
            YuzuhaError::InvalidContraction(msg) => {
                PyValueError::new_err(format!("Invalid contraction: {}", msg))
            }
            YuzuhaError::CacheError(msg) => {
                PyRuntimeError::new_err(format!("Cache error: {}", msg))
            }
            _ => {
                PyRuntimeError::new_err(format!("Yuzuha error: {}", err))
            }
        }
    }
}

/// Python wrapper for Spin quantum number
///
/// Represents a spin quantum number stored as a doubled integer (2j).
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> j_half = yuzuha.Spin(1)  # j=1/2
/// >>> print(j_half.value())
/// 0.5
/// >>> j1 = yuzuha.Spin(2)  # j=1
/// >>> print(j1.dimension())
/// 3
#[pyclass(name = "Spin")]
#[derive(Clone)]
pub struct PySpin {
    inner: RustSpin,
}

#[pymethods]
impl PySpin {
    /// Create a new Spin from a doubled integer value.
    ///
    /// Parameters
    /// ----------
    /// j_doubled : int
    ///     The doubled spin value (2j). Must be non-negative.
    ///
    /// Returns
    /// -------
    /// Spin
    ///     A new Spin object.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If j_doubled is negative.
    #[new]
    fn new(j_doubled: i32) -> PyResult<Self> {
        Ok(PySpin {
            inner: RustSpin::new(j_doubled)?,
        })
    }

    /// Get the floating-point value of the spin (j).
    ///
    /// Returns
    /// -------
    /// float
    ///     The spin value j = j_doubled / 2.
    fn value(&self) -> f64 {
        self.inner.value()
    }

    /// Get the doubled integer value (2j).
    ///
    /// Returns
    /// -------
    /// int
    ///     The doubled spin value.
    fn twice(&self) -> i32 {
        self.inner.twice()
    }

    /// Get the dimension of this spin representation (2j + 1).
    ///
    /// Returns
    /// -------
    /// int
    ///     The dimension 2j + 1.
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Check if this is an integer spin.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if j is an integer, False if half-integer.
    fn is_integer(&self) -> bool {
        self.inner.is_integer()
    }

    /// Check if this is a half-integer spin.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if j is a half-integer, False if integer.
    fn is_half_integer(&self) -> bool {
        self.inner.is_half_integer()
    }

    fn __repr__(&self) -> String {
        format!("Spin({})", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Python wrapper for Direction (edge orientation)
///
/// Represents the orientation of a tensor edge: incoming (+1) or outgoing (-1).
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> d = yuzuha.Direction.incoming()
/// >>> print(d.sign())
/// 1
/// >>> print(d.flip())
/// Direction.outgoing
#[pyclass(name = "Direction")]
#[derive(Clone)]
pub struct PyDirection {
    inner: Direction,
}

#[pymethods]
impl PyDirection {
    /// Create an incoming Direction.
    ///
    /// Returns
    /// -------
    /// Direction
    ///     An incoming direction (sign = +1).
    #[staticmethod]
    fn incoming() -> Self {
        PyDirection { inner: Direction::Incoming }
    }

    /// Create an outgoing Direction.
    ///
    /// Returns
    /// -------
    /// Direction
    ///     An outgoing direction (sign = -1).
    #[staticmethod]
    fn outgoing() -> Self {
        PyDirection { inner: Direction::Outgoing }
    }

    /// Create a Direction from its sign value (+1 or -1).
    ///
    /// Parameters
    /// ----------
    /// sign : int
    ///     +1 for incoming, -1 for outgoing.
    ///
    /// Returns
    /// -------
    /// Direction
    ///     The corresponding Direction.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If sign is not +1 or -1.
    #[staticmethod]
    fn from_sign(sign: i32) -> PyResult<Self> {
        Ok(PyDirection {
            inner: Direction::from_sign(sign)?,
        })
    }

    /// Get the numerical sign of this direction.
    ///
    /// Returns
    /// -------
    /// int
    ///     +1 for incoming, -1 for outgoing.
    fn sign(&self) -> i32 {
        self.inner.sign()
    }

    /// Return the flipped Direction.
    ///
    /// Returns
    /// -------
    /// Direction
    ///     The opposite direction.
    fn flip(&self) -> Self {
        PyDirection { inner: self.inner.flip() }
    }

    /// Check if this direction is incoming.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if incoming.
    fn is_incoming(&self) -> bool {
        self.inner == Direction::Incoming
    }

    /// Check if this direction is outgoing.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if outgoing.
    fn is_outgoing(&self) -> bool {
        self.inner == Direction::Outgoing
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Direction::Incoming => "Direction.incoming".to_string(),
            Direction::Outgoing => "Direction.outgoing".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &PyDirection) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> i32 {
        self.inner.sign()
    }
}

/// Python wrapper for Edge (tensor index)
///
/// Represents a tensor edge with a spin quantum number and direction.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> j = yuzuha.Spin(1)
/// >>> edge_in = yuzuha.Edge.incoming(j)
/// >>> edge_out = yuzuha.Edge.outgoing(j)
#[pyclass(name = "Edge")]
#[derive(Clone)]
pub struct PyEdge {
    inner: RustEdge,
}

#[pymethods]
impl PyEdge {
    /// Create an incoming edge with the given spin.
    ///
    /// Parameters
    /// ----------
    /// spin : Spin
    ///     The spin quantum number for this edge.
    ///
    /// Returns
    /// -------
    /// Edge
    ///     An incoming edge.
    #[staticmethod]
    fn incoming(spin: &PySpin) -> Self {
        PyEdge {
            inner: RustEdge::incoming(spin.inner),
        }
    }

    /// Create an outgoing edge with the given spin.
    ///
    /// Parameters
    /// ----------
    /// spin : Spin
    ///     The spin quantum number for this edge.
    ///
    /// Returns
    /// -------
    /// Edge
    ///     An outgoing edge.
    #[staticmethod]
    fn outgoing(spin: &PySpin) -> Self {
        PyEdge {
            inner: RustEdge::outgoing(spin.inner),
        }
    }

    /// Get the spin of this edge.
    ///
    /// Returns
    /// -------
    /// Spin
    ///     The spin quantum number.
    #[getter]
    fn j(&self) -> PySpin {
        PySpin {
            inner: self.inner.j,
        }
    }

    /// Get the direction of this edge.
    ///
    /// Returns
    /// -------
    /// Direction
    ///     The edge direction (incoming or outgoing).
    #[getter]
    fn dir(&self) -> PyDirection {
        PyDirection { inner: self.inner.dir }
    }

    /// Check if this edge is incoming.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if incoming, False if outgoing.
    fn is_incoming(&self) -> bool {
        self.inner.dir == Direction::Incoming
    }

    /// Check if this edge is outgoing.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if outgoing, False if incoming.
    fn is_outgoing(&self) -> bool {
        self.inner.dir == Direction::Outgoing
    }

    fn __repr__(&self) -> String {
        let dir = if self.inner.dir == Direction::Incoming {
            "incoming"
        } else {
            "outgoing"
        };
        format!("Edge.{}(Spin({}))", dir, self.inner.j)
    }
}

/// Python wrapper for CGSpec (Coupled Gauge Tree Specification)
///
/// Represents a coupled gauge tree with external edges and outer multiplicity configurations.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> j = yuzuha.Spin(1)
/// >>> edges = [yuzuha.Edge.incoming(j), yuzuha.Edge.incoming(j), yuzuha.Edge.outgoing(j)]
/// >>> spec = yuzuha.CGSpec.from_edges(edges)
/// >>> print(spec.num_external())
/// 3
/// >>> print(spec.om_dimension())
/// 1
#[pyclass(name = "CGSpec")]
#[derive(Clone)]
pub struct PyCGSpec {
    inner: RustCGSpec,
}

#[pymethods]
impl PyCGSpec {
    /// Create a CGSpec from a list of edges.
    ///
    /// Parameters
    /// ----------
    /// edges : list[Edge]
    ///     List of edges defining the tensor structure.
    ///
    /// Returns
    /// -------
    /// CGSpec
    ///     A new CGSpec with automatically enumerated OM configurations.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the edge configuration is invalid.
    #[staticmethod]
    fn from_edges(edges: Vec<PyRef<PyEdge>>) -> PyResult<Self> {
        let rust_edges: Vec<RustEdge> = edges.iter().map(|e| e.inner.clone()).collect();
        Ok(PyCGSpec {
            inner: RustCGSpec::from_edges(rust_edges)?,
        })
    }

    /// Get the number of external edges.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of external edges.
    fn num_external(&self) -> usize {
        self.inner.num_external()
    }

    /// Get the outer multiplicity dimension.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of distinct OM configurations.
    fn om_dimension(&self) -> usize {
        self.inner.om_dimension()
    }

    /// Get the edges of this CGSpec.
    ///
    /// Returns
    /// -------
    /// list[Edge]
    ///     List of edges defining the tensor structure.
    #[getter]
    fn edges(&self) -> Vec<PyEdge> {
        self.inner.edges.iter().map(|e| PyEdge { inner: e.clone() }).collect()
    }

    /// Get the spin values (doubled) for all edges.
    ///
    /// Returns
    /// -------
    /// list[int]
    ///     List of doubled spin values (2j) for each edge.
    fn get_spins(&self) -> Vec<i32> {
        self.inner.edges.iter().map(|e| e.j.twice()).collect()
    }

    /// Get the directions for all edges.
    ///
    /// Returns
    /// -------
    /// list[int]
    ///     List of directions: +1 for incoming, -1 for outgoing.
    fn get_directions(&self) -> Vec<i8> {
        self.inner.edges.iter().map(|e| match e.dir {
            Direction::Incoming => 1,
            Direction::Outgoing => -1,
        }).collect()
    }

    /// Return a new CGSpec with the edge directions at the given axes flipped.
    ///
    /// The OM configurations are reused unchanged because they depend only on
    /// spin values, not directions.
    ///
    /// Parameters
    /// ----------
    /// axes : list[int]
    ///     Indices of edges whose direction should be flipped.
    ///
    /// Returns
    /// -------
    /// CGSpec
    ///     A new CGSpec with the specified edge directions inverted.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any axis index is out of bounds.
    ///
    /// Examples
    /// --------
    /// >>> import yuzuha
    /// >>> j = yuzuha.Spin(2)
    /// >>> spec = yuzuha.CGSpec.from_edges([
    /// ...     yuzuha.Edge.incoming(j),
    /// ...     yuzuha.Edge.incoming(j),
    /// ...     yuzuha.Edge.outgoing(j),
    /// ... ])
    /// >>> flipped = spec.with_inverted_axes([0, 1])
    /// >>> [e.dir for e in flipped.edges]
    /// [Direction.outgoing, Direction.outgoing, Direction.outgoing]
    fn with_inverted_axes(&self, axes: Vec<usize>) -> PyResult<Self> {
        Ok(PyCGSpec {
            inner: self.inner.with_inverted_axes(&axes)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("CGSpec(num_external={}, om_dim={})", 
                self.inner.num_external(), 
                self.inner.om_dimension())
    }
}

/// Python wrapper for Contraction specification
///
/// Specifies which edges to contract between two CGSpecs.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> # Contract edge 2 from A with edge 0 from B
/// >>> contraction = yuzuha.Contraction([2], [0])
#[pyclass(name = "Contraction")]
#[derive(Clone)]
pub struct PyContraction {
    inner: RustContraction,
}

#[pymethods]
impl PyContraction {
    /// Create a new Contraction specification.
    ///
    /// Parameters
    /// ----------
    /// axes_a : list[int]
    ///     Indices of edges to contract from the first CGSpec.
    /// axes_b : list[int]
    ///     Indices of edges to contract from the second CGSpec.
    ///
    /// Returns
    /// -------
    /// Contraction
    ///     A new Contraction specification.
    #[new]
    fn new(axes_a: Vec<usize>, axes_b: Vec<usize>) -> Self {
        PyContraction {
            inner: RustContraction::new(&axes_a, &axes_b),
        }
    }

    /// Get the axes from the first tensor.
    ///
    /// Returns
    /// -------
    /// list[int]
    ///     Indices of edges to contract from the first CGSpec.
    #[getter]
    fn axes_a(&self) -> Vec<usize> {
        self.inner.axes_a.clone()
    }

    /// Get the axes from the second tensor.
    ///
    /// Returns
    /// -------
    /// list[int]
    ///     Indices of edges to contract from the second CGSpec.
    #[getter]
    fn axes_b(&self) -> Vec<usize> {
        self.inner.axes_b.clone()
    }

    fn __repr__(&self) -> String {
        format!("Contraction(axes_a={:?}, axes_b={:?})", 
                self.inner.axes_a, 
                self.inner.axes_b)
    }
}

/// Compute X-symbol for tensor network contraction.
///
/// The X-symbol represents the coupling coefficients for contracting two
/// coupled gauge trees (CGTs) with specified edges.
///
/// Parameters
/// ----------
/// spec_a : CGSpec
///     First CGSpec to contract.
/// spec_b : CGSpec
///     Second CGSpec to contract.
/// contraction : Contraction
///     Specification of which edges to contract.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, CGSpec]
///     A tuple containing:
///     - X-symbol array of shape [om_a, om_b, om_c] where om_a, om_b, om_c
///       are the outer multiplicity dimensions of A, B, and C respectively.
///     - Output CGSpec C representing the contracted tensor structure.
///
/// Raises
/// ------
/// ValueError
///     If the contraction specification is invalid.
/// RuntimeError
///     If computation fails or cache error occurs.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> j_half = yuzuha.Spin(1)
/// >>> spec_a = yuzuha.CGSpec.from_edges([
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.outgoing(yuzuha.Spin(2))
/// ... ])
/// >>> spec_b = yuzuha.CGSpec.from_edges([
/// ...     yuzuha.Edge.incoming(yuzuha.Spin(2)),
/// ...     yuzuha.Edge.outgoing(j_half),
/// ...     yuzuha.Edge.outgoing(j_half)
/// ... ])
/// >>> contraction = yuzuha.Contraction([2], [0])
/// >>> x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
/// >>> print(x_array.shape)
/// (1, 1, 2)
#[pyfunction]
fn compute_xsymbol<'py>(
    py: Python<'py>,
    spec_a: &PyCGSpec,
    spec_b: &PyCGSpec,
    contraction: &PyContraction,
) -> PyResult<(Bound<'py, PyArray3<f64>>, PyCGSpec)> {
    // Compute the X-symbol
    let result = rust_compute_xsymbol(&spec_a.inner, &spec_b.inner, &contraction.inner)?;
    
    // Convert ndarray to numpy array
    let x_array = PyArray3::from_owned_array(py, result.data);
    let spec_c = PyCGSpec { inner: result.spec_c };
    
    Ok((x_array, spec_c))
}

/// Compute R-symbol for tensor edge permutation.
///
/// The R-symbol represents how outer multiplicity indices transform under
/// permutation of tensor edges.
///
/// Parameters
/// ----------
/// spec : CGSpec
///     The CGSpec to permute.
/// permutation : list[int]
///     Permutation of external edge indices. Must be a valid permutation
///     (each index from 0 to num_external-1 appears exactly once).
///
/// Returns
/// -------
/// tuple[numpy.ndarray, CGSpec]
///     A tuple containing:
///     - R-symbol array of shape [om_original, om_permuted].
///     - Permuted CGSpec with edges reordered according to permutation.
///
/// Raises
/// ------
/// ValueError
///     If the permutation is invalid.
/// RuntimeError
///     If computation fails or cache error occurs.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> j_half = yuzuha.Spin(1)
/// >>> spec = yuzuha.CGSpec.from_edges([
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half)
/// ... ])
/// >>> # Swap first two edges
/// >>> permutation = [1, 0, 2, 3]
/// >>> r_array, spec_permuted = yuzuha.compute_rsymbol(spec, permutation)
/// >>> print(r_array.shape)
/// (2, 2)
#[pyfunction]
fn compute_rsymbol<'py>(
    py: Python<'py>,
    spec: &PyCGSpec,
    permutation: Vec<usize>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, PyCGSpec)> {
    // Compute the R-symbol
    let result = rust_compute_rsymbol(&spec.inner, &permutation)?;
    
    // Convert ndarray to numpy array
    let r_array = PyArray2::from_owned_array(py, result.data);
    let spec_permuted = PyCGSpec { inner: result.spec_permuted };
    
    Ok((r_array, spec_permuted))
}

/// Compute canonical basis data for a coupled gauge tree.
///
/// Computes the transformation matrix from magnetic quantum number basis to
/// outer multiplicity (OM) basis for a given CGSpec. The basis is cached for
/// efficiency using an SQLite database.
///
/// The canonical basis is only defined for CGSpecs with at least 3 external edges.
///
/// Parameters
/// ----------
/// spec : CGSpec
///     The CGSpec for which to compute the canonical basis.
///
/// Returns
/// -------
/// numpy.ndarray
///     The canonical basis transformation matrix. Shape depends on the CGSpec:
///     - For n external edges with spins j_i, shape is [d_1, d_2, ..., d_n, om_dim]
///       where d_i = 2*j_i + 1 is the dimension of the i-th edge, and om_dim
///       is the outer multiplicity dimension.
///
/// Raises
/// ------
/// ValueError
///     If the CGSpec has fewer than 3 external edges.
/// RuntimeError
///     If computation fails or cache error occurs.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> j_half = yuzuha.Spin(1)  # j=1/2
/// >>> spec = yuzuha.CGSpec.from_edges([
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half)
/// ... ])
/// >>> basis = yuzuha.canonical_basis(spec)
/// >>> print(basis.shape)
/// (2, 2, 2, 2)
#[pyfunction]
fn canonical_basis<'py>(
    py: Python<'py>,
    spec: &PyCGSpec,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    // Compute the canonical basis data
    let result = rust_build_canonical_basis_data(&spec.inner)?;
    
    // Convert ndarray to numpy array
    let basis_array = PyArrayDyn::from_owned_array(py, result);
    
    Ok(basis_array)
}

/// Return the Frobenius-Schur phase ``(-1)^{2j}`` for a single spin.
///
/// Parameters
/// ----------
/// spin : Spin
///     The spin quantum number.
///
/// Returns
/// -------
/// float
///     ``+1.0`` for integer spins, ``-1.0`` for half-integer spins.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> yuzuha.fs_phase_for_spin(yuzuha.Spin(1))   # j=1/2, half-integer
/// -1.0
/// >>> yuzuha.fs_phase_for_spin(yuzuha.Spin(2))   # j=1, integer
/// 1.0
#[pyfunction]
fn fs_phase_for_spin(spin: &PySpin) -> f64 {
    rust_fs_phase_for_spin(spin.inner)
}

/// Compute the conjugate CGSpec and the cumulated Frobenius-Schur phase.
///
/// The conjugated spec has all edge directions reversed. The cumulated FS
/// phase corrects for edges in the conjugated spec that differ from the
/// canonical conjugate pattern (first ``n-1`` edges Outgoing, last edge
/// Incoming). Each such differing edge contributes a factor of ``(-1)^{2j}``.
///
/// Parameters
/// ----------
/// spec : CGSpec
///     The CGSpec to conjugate.
///
/// Returns
/// -------
/// tuple[float, CGSpec]
///     A tuple ``(phase, conj_spec)`` where ``phase`` is ``+1.0`` or ``-1.0``
///     and ``conj_spec`` has the same spins as ``spec`` with all directions
///     flipped.
///
/// Raises
/// ------
/// ValueError
///     If the conjugated edge configuration is invalid.
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> j_half = yuzuha.Spin(1)
/// >>> spec = yuzuha.CGSpec.from_edges([
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.outgoing(yuzuha.Spin(2)),
/// ... ])
/// >>> phase, conj = yuzuha.compute_conjugate(spec)
/// >>> phase
/// 1.0
#[pyfunction]
fn compute_conjugate(spec: &PyCGSpec) -> PyResult<(f64, PyCGSpec)> {
    let (phase, conj_spec) = rust_compute_conjugate(&spec.inner)?;
    Ok((phase, PyCGSpec { inner: conj_spec }))
}

/// Yuzuha: SU(2) X-symbols for Tensor Networks
///
/// A library for computing SU(2) X-symbols (recoupling coefficients) for
/// arbitrary tensor network contractions using left-associative fusion trees.
///
/// Features
/// --------
/// - Compute X-symbols for arbitrary tensor network contractions
/// - Compute R-symbols for tensor edge permutations
/// - Automatic caching of computed basis data using SQLite
/// - Thread-safe database connection management
/// - Real-valued Condon-Shortley convention for CG coefficients
///
/// Configuration
/// -------------
/// Set the YUZUHA_CACHE_PATH environment variable to specify the cache
/// directory. All cache databases will be stored in this directory:
/// - cgbasis.db (canonical basis cache)
/// - xsymbol.db (X-symbol cache)
/// - rsymbol.db (R-symbol cache)
/// Default directory: .yuzuha/
///
/// Examples
/// --------
/// >>> import yuzuha
/// >>> # Create spins
/// >>> j_half = yuzuha.Spin(1)  # j=1/2
/// >>> j1 = yuzuha.Spin(2)      # j=1
/// >>>
/// >>> # Create edges
/// >>> edges_a = [
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.incoming(j_half),
/// ...     yuzuha.Edge.outgoing(j1)
/// ... ]
/// >>> spec_a = yuzuha.CGSpec.from_edges(edges_a)
/// >>>
/// >>> # Compute X-symbol
/// >>> contraction = yuzuha.Contraction([2], [0])
/// >>> x_array, spec_c = yuzuha.compute_xsymbol(spec_a, spec_b, contraction)
#[pymodule]
fn yuzuha(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDirection>()?;
    m.add_class::<PySpin>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyCGSpec>()?;
    m.add_class::<PyContraction>()?;
    m.add_function(wrap_pyfunction!(compute_xsymbol, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rsymbol, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_basis, m)?)?;
    m.add_function(wrap_pyfunction!(fs_phase_for_spin, m)?)?;
    m.add_function(wrap_pyfunction!(compute_conjugate, m)?)?;
    Ok(())
}
