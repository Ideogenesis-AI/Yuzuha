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

//! Core data structures for SU(2) representation theory
//!
//! This module provides fundamental types for working with SU(2) spins,
//! coupled gauge trees (CGTs), and tensor network contractions.

use crate::error::{Result, YuzuhaError};
use std::fmt;

/// Spin quantum number stored as a doubled integer (J = 2j)
///
/// Using doubled integers avoids floating-point arithmetic:
/// - j = 1/2 is represented as J = 1
/// - j = 1 is represented as J = 2
/// - j = 3/2 is represented as J = 3
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Spin(i32);

impl Spin {
    /// Create a new spin quantum number from a doubled integer
    ///
    /// # Arguments
    /// * `j_doubled` - The doubled spin value (2j)
    ///
    /// # Errors
    /// Returns an error if the value is negative
    pub fn new(j_doubled: i32) -> Result<Self> {
        if j_doubled < 0 {
            return Err(YuzuhaError::InvalidSpin(j_doubled));
        }
        Ok(Spin(j_doubled))
    }

    /// Get the doubled value (2j)
    #[inline]
    pub fn twice(&self) -> i32 {
        self.0
    }

    /// Get the floating-point value of j
    #[inline]
    pub fn value(&self) -> f64 {
        self.0 as f64 / 2.0
    }

    /// Get the dimension of this spin representation (2j + 1)
    #[inline]
    pub fn dimension(&self) -> usize {
        (self.0 + 1) as usize
    }

    /// Check if this is an integer spin (j is integer)
    #[inline]
    pub fn is_integer(&self) -> bool {
        self.0 % 2 == 0
    }

    /// Check if this is a half-integer spin (j is half-integer)
    #[inline]
    pub fn is_half_integer(&self) -> bool {
        self.0 % 2 == 1
    }
}

impl fmt::Display for Spin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 % 2 == 0 {
            write!(f, "{}", self.0 / 2)
        } else {
            write!(f, "{}/2", self.0)
        }
    }
}

/// Magnetic quantum number stored as a doubled integer (M = 2m)
///
/// For a given spin J, valid values are M ∈ {-J, -J+2, ..., J-2, J}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MagneticNumber(i32);

impl MagneticNumber {
    /// Create a new magnetic quantum number
    ///
    /// # Arguments
    /// * `m_doubled` - The doubled magnetic quantum number (2m)
    /// * `j` - The associated spin quantum number for validation
    ///
    /// # Errors
    /// Returns an error if m is out of range for the given j
    pub fn new(m_doubled: i32, j: Spin) -> Result<Self> {
        let j_val = j.twice();
        if m_doubled < -j_val || m_doubled > j_val || (m_doubled - j_val) % 2 != 0 {
            return Err(YuzuhaError::InvalidMagneticNumber {
                j: j_val,
                m: m_doubled,
            });
        }
        Ok(MagneticNumber(m_doubled))
    }

    /// Create without validation (use carefully!)
    #[inline]
    pub fn new_unchecked(m_doubled: i32) -> Self {
        MagneticNumber(m_doubled)
    }

    /// Get the doubled value (2m)
    #[inline]
    pub fn twice(&self) -> i32 {
        self.0
    }

    /// Get the floating-point value of m
    #[inline]
    pub fn value(&self) -> f64 {
        self.0 as f64 / 2.0
    }

    /// Negate the magnetic quantum number
    #[inline]
    pub fn negate(&self) -> Self {
        MagneticNumber(-self.0)
    }

    /// Convert to array index for a given spin
    /// Returns (M + J) / 2
    #[inline]
    pub fn to_index(&self, j: Spin) -> usize {
        ((self.0 + j.twice()) / 2) as usize
    }

    /// Create from array index for a given spin
    /// M = -J + 2*index
    #[inline]
    pub fn from_index(index: usize, j: Spin) -> Self {
        MagneticNumber(-j.twice() + 2 * index as i32)
    }
}

impl std::ops::Add for MagneticNumber {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        MagneticNumber(self.0 + rhs.0)
    }
}

impl fmt::Display for MagneticNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 % 2 == 0 {
            write!(f, "{}", self.0 / 2)
        } else {
            write!(f, "{}/2", self.0)
        }
    }
}

/// Arrow direction for tensor legs
///
/// In tensor category theory:
/// - `Incoming` (+1): leg points into the tensor
/// - `Outgoing` (-1): leg points out of the tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Incoming leg (arrow points in)
    Incoming,
    /// Outgoing leg (arrow points out)
    Outgoing,
}

impl Direction {
    /// Get the numerical sign (+1 or -1)
    #[inline]
    pub fn sign(&self) -> i32 {
        match self {
            Direction::Incoming => 1,
            Direction::Outgoing => -1,
        }
    }

    /// Flip the direction
    #[inline]
    pub fn flip(&self) -> Self {
        match self {
            Direction::Incoming => Direction::Outgoing,
            Direction::Outgoing => Direction::Incoming,
        }
    }

    /// Create from sign value
    pub fn from_sign(sign: i32) -> Result<Self> {
        match sign {
            1 => Ok(Direction::Incoming),
            -1 => Ok(Direction::Outgoing),
            _ => Err(YuzuhaError::InvalidCGTSpec(format!(
                "Invalid direction sign: {}",
                sign
            ))),
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Incoming => write!(f, "→"),
            Direction::Outgoing => write!(f, "←"),
        }
    }
}

/// Edge of a tensor in the category-theoretic sense
///
/// Each edge has a spin quantum number and arrow direction.
/// This is the unified concept that replaces the old "Leg" terminology.
/// Edges are identified by their position/index in the edge list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    /// Spin quantum number (doubled)
    pub j: Spin,
    /// Arrow direction
    pub dir: Direction,
}

impl Edge {
    /// Create a new edge
    pub fn new(j: Spin, dir: Direction) -> Self {
        Edge { j, dir }
    }

    /// Create with incoming direction
    pub fn incoming(j: Spin) -> Self {
        Self::new(j, Direction::Incoming)
    }

    /// Create with outgoing direction
    pub fn outgoing(j: Spin) -> Self {
        Self::new(j, Direction::Outgoing)
    }

    /// Flip the arrow direction
    pub fn with_flipped_direction(&self) -> Self {
        Edge {
            j: self.j,
            dir: self.dir.flip(),
        }
    }

    /// Get the dimension of this edge (2j+1)
    #[inline]
    pub fn dimension(&self) -> usize {
        self.j.dimension()
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[j={}, {}]", self.j, self.dir)
    }
}

/// Coupled Gauge (CG) Specification
///
/// Represents the topology and orthonormal multiplicity (OM) structure
/// for a tensor with multiple external edges.
///
/// Unlike the old CGTSpec which held a single OM configuration,
/// CGSpec encompasses ALL valid OM configurations (alphas) for the
/// given external edge structure.
///
/// Uses deterministic Condon-Shortley convention for CG coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CGSpec {
    /// Ordered external edges
    pub edges: Vec<Edge>,
    /// ALL valid internal spin configurations (OM basis)
    /// Each alpha is a vector of internal spins for a left-associative fusion tree
    pub alphas: Vec<Vec<Spin>>,
}

impl CGSpec {
    /// Create a new CG specification with given edges and OM configurations
    ///
    /// # Arguments
    /// * `edges` - External edges in fusion order
    /// * `alphas` - All valid OM configurations
    pub fn new(edges: Vec<Edge>, alphas: Vec<Vec<Spin>>) -> Result<Self> {
        let n = edges.len();
        let expected_alpha_len = if n >= 3 { n - 2 } else { 0 };

        for alpha in &alphas {
            if alpha.len() != expected_alpha_len {
                return Err(YuzuhaError::InvalidCGTSpec(format!(
                    "Expected {} internal spins for {} edges, got {}",
                    expected_alpha_len,
                    n,
                    alpha.len()
                )));
            }
        }

        Ok(CGSpec {
            edges,
            alphas,
        })
    }

    /// Create from edges, automatically enumerating all valid OM configurations
    ///
    /// This is the primary constructor - it computes all valid fusion trees
    /// for the given external edges using the Condon-Shortley convention.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge spins cannot satisfy SU(2) angular momentum
    /// conservation (i.e., cannot couple to total j=0). This can happen for:
    /// - Odd number of spin-1/2 particles (fermionic parity violation)
    /// - Edge configurations that violate triangle inequalities
    pub fn from_edges(edges: Vec<Edge>) -> Result<Self> {
        let j_list: Vec<Spin> = edges.iter().map(|e| e.j).collect();
        let alphas = crate::builders::om_basis::enumerate_alpha(&j_list);
        
        if alphas.is_empty() {
            return Err(YuzuhaError::InvalidCGTSpec(
                format!(
                    "Edge spins {:?} do not satisfy SU(2) angular momentum conservation. \
                     Cannot construct valid fusion tree coupling to j=0. \
                     Common causes: odd number of fermions (j=1/2) or incompatible spin values.",
                    j_list.iter().map(|j| format!("j={}/{}", j.twice(), 2)).collect::<Vec<_>>()
                )
            ));
        }
        
        Ok(CGSpec {
            edges,
            alphas,
        })
    }

    /// Get the number of external edges
    #[inline]
    pub fn num_external(&self) -> usize {
        self.edges.len()
    }

    /// Get the orthonormal multiplicity dimension (number of valid OM configurations)
    #[inline]
    pub fn om_dimension(&self) -> usize {
        self.alphas.len()
    }

    /// Get the full tensor shape: [external_dims..., om_dim]
    pub fn shape(&self) -> Vec<usize> {
        let mut shape: Vec<usize> = self.edges.iter()
            .map(|e| e.dimension())
            .collect();
        shape.push(self.om_dimension());
        shape
    }

    /// Get the spin of a specific edge by index
    pub fn edge_spin_at(&self, idx: usize) -> Result<Spin> {
        self.edges
            .get(idx)
            .map(|edge| edge.j)
            .ok_or_else(|| YuzuhaError::IndexOutOfBounds(idx, self.edges.len()))
    }

}

/// Coupled Gauge Tensor
///
/// A tensor over external indices and OM configurations.
/// The data array has shape [external_dims..., om_dim] where the last
/// axis corresponds to the orthonormal multiplicity.
#[derive(Debug, Clone)]
pub struct CGTensor {
    /// Specification (topology and OM structure)
    pub spec: CGSpec,
    /// Tensor data: shape = [external_dims..., om_dim]
    pub data: ndarray::ArrayD<f64>,
}

impl CGTensor {
    /// Create a zero tensor from a specification
    pub fn zeros(spec: CGSpec) -> Self {
        let shape = spec.shape();
        let data = ndarray::ArrayD::zeros(ndarray::IxDyn(&shape));
        CGTensor { spec, data }
    }

    /// Create a tensor from spec and data
    ///
    /// # Errors
    /// Returns error if data shape doesn't match spec shape
    pub fn new(spec: CGSpec, data: ndarray::ArrayD<f64>) -> Result<Self> {
        let expected_shape = spec.shape();
        let actual_shape = data.shape();
        
        if expected_shape.as_slice() != actual_shape {
            return Err(YuzuhaError::DimensionMismatch {
                expected: expected_shape.len(),
                actual: actual_shape.len(),
            });
        }

        Ok(CGTensor { spec, data })
    }

    /// Get the number of external edges
    #[inline]
    pub fn num_external(&self) -> usize {
        self.spec.num_external()
    }

    /// Get the OM dimension
    #[inline]
    pub fn om_dimension(&self) -> usize {
        self.spec.om_dimension()
    }

    /// Get a view of data for a specific OM configuration
    ///
    /// Returns a view of shape [external_dims...]
    pub fn om_slice(&self, om_idx: usize) -> Result<ndarray::ArrayViewD<'_, f64>> {
        if om_idx >= self.om_dimension() {
            return Err(YuzuhaError::InvalidCGTSpec(format!(
                "OM index {} out of range (max {})",
                om_idx,
                self.om_dimension() - 1
            )));
        }

        let n_external = self.num_external();
        let mut slice_info = vec![ndarray::SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        }; n_external];
        slice_info.push(ndarray::SliceInfoElem::Index(om_idx as isize));

        Ok(self.data.slice(ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::try_from(slice_info).unwrap()))
    }

    /// Set data for a specific OM configuration
    pub fn set_om_slice(&mut self, om_idx: usize, slice_data: ndarray::ArrayViewD<f64>) -> Result<()> {
        if om_idx >= self.om_dimension() {
            return Err(YuzuhaError::InvalidCGTSpec(format!(
                "OM index {} out of range (max {})",
                om_idx,
                self.om_dimension() - 1
            )));
        }

        let n_external = self.num_external();
        let mut slice_info = vec![ndarray::SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        }; n_external];
        slice_info.push(ndarray::SliceInfoElem::Index(om_idx as isize));

        let mut view = self.data.slice_mut(ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::try_from(slice_info).unwrap());
        view.assign(&slice_data);

        Ok(())
    }

    /// Create a canonical basis tensor from edges
    ///
    /// Automatically enumerates OM configurations and computes the canonical
    /// basis for the given external edges using SU(2) fusion rules.
    ///
    /// # Arguments
    /// * `edges` - External edges in fusion order
    ///
    /// # Returns
    /// CGTensor containing the canonical basis data
    pub fn canonical_basis_from_edges(edges: Vec<Edge>) -> Result<Self> {
        let spec = CGSpec::from_edges(edges)?;
        let data = crate::builders::builders::build_canonical_basis_data(&spec)?;
        Ok(CGTensor { spec, data })
    }

    /// Create a canonical basis tensor from an existing CGSpec
    ///
    /// Computes the canonical basis for all OM configurations in the spec.
    ///
    /// # Arguments
    /// * `spec` - CG specification with edges and alphas
    ///
    /// # Returns
    /// CGTensor containing the canonical basis data
    pub fn canonical_basis_from_spec(spec: CGSpec) -> Result<Self> {
        let data = crate::builders::builders::build_canonical_basis_data(&spec)?;
        Ok(CGTensor { spec, data })
    }
}

/// Specification for contracting two CGTs
///
/// Defines which edges from CGT A connect to which edges from CGT B.
/// Similar to numpy's tensordot axes parameter.
///
/// # Examples
///
/// Single edge contraction:
/// ```
/// # use yuzuha::Contraction;
/// // Contract edge 1 from A with edge 0 from B
/// let contraction = Contraction::new(&[1], &[0]);
/// ```
///
/// Multiple edge contraction (like numpy.tensordot):
/// ```
/// # use yuzuha::Contraction;
/// // Contract edges 1,2 from A with edges 0,3 from B
/// // Similar to: np.tensordot(A, B, axes=([1, 2], [0, 3]))
/// let contraction = Contraction::new(&[1, 2], &[0, 3]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contraction {
    /// Edge indices from tensor A to contract
    pub axes_a: Vec<usize>,
    /// Edge indices from tensor B to contract (must have same length as axes_a)
    pub axes_b: Vec<usize>,
}

impl Contraction {
    /// Create a new contraction specification (numpy tensordot style)
    ///
    /// # Arguments
    /// * `axes_a` - Edge indices from tensor A
    /// * `axes_b` - Edge indices from tensor B
    ///
    /// # Panics
    /// Panics if the two arrays have different lengths.
    pub fn new(axes_a: &[usize], axes_b: &[usize]) -> Self {
        assert_eq!(
            axes_a.len(),
            axes_b.len(),
            "Contraction axes must have same length"
        );
        Contraction {
            axes_a: axes_a.to_vec(),
            axes_b: axes_b.to_vec(),
        }
    }

    /// Create an empty contraction (no edges contracted)
    pub fn empty() -> Self {
        Contraction {
            axes_a: Vec::new(),
            axes_b: Vec::new(),
        }
    }

    /// Check if this is an empty contraction
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.axes_a.is_empty()
    }

    /// Get the number of contracted edge pairs
    #[inline]
    pub fn num_pairs(&self) -> usize {
        self.axes_a.len()
    }

    /// Validate that the contraction is compatible with two CG specs
    pub fn validate(&self, spec_a: &CGSpec, spec_b: &CGSpec) -> Result<()> {
        for (&idx_a, &idx_b) in self.axes_a.iter().zip(self.axes_b.iter()) {
            let spin_a = spec_a.edge_spin_at(idx_a)?;
            let spin_b = spec_b.edge_spin_at(idx_b)?;

            if spin_a != spin_b {
                return Err(YuzuhaError::IncompatibleLegs(
                    spin_a.twice(),
                    spin_b.twice(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_basic() {
        let j0 = Spin::new(0).unwrap();
        assert_eq!(j0.twice(), 0);
        assert_eq!(j0.value(), 0.0);
        assert_eq!(j0.dimension(), 1);
        assert!(j0.is_integer());

        let j_half = Spin::new(1).unwrap();
        assert_eq!(j_half.twice(), 1);
        assert_eq!(j_half.value(), 0.5);
        assert_eq!(j_half.dimension(), 2);
        assert!(j_half.is_half_integer());

        let j1 = Spin::new(2).unwrap();
        assert_eq!(j1.twice(), 2);
        assert_eq!(j1.value(), 1.0);
        assert_eq!(j1.dimension(), 3);

        assert!(Spin::new(-1).is_err());
    }

    #[test]
    fn test_magnetic_number() {
        let j1 = Spin::new(2).unwrap();

        let m = MagneticNumber::new(2, j1).unwrap();
        assert_eq!(m.twice(), 2);
        assert_eq!(m.value(), 1.0);

        let m_neg = m.negate();
        assert_eq!(m_neg.twice(), -2);

        // Test index conversion
        assert_eq!(m.to_index(j1), 2);
        assert_eq!(MagneticNumber::from_index(2, j1).twice(), 2);

        // Test out of range
        assert!(MagneticNumber::new(4, j1).is_err());
        assert!(MagneticNumber::new(1, j1).is_err()); // Wrong parity
    }

    #[test]
    fn test_direction() {
        let dir_in = Direction::Incoming;
        let dir_out = Direction::Outgoing;

        assert_eq!(dir_in.sign(), 1);
        assert_eq!(dir_out.sign(), -1);

        assert_eq!(dir_in.flip(), dir_out);
        assert_eq!(dir_out.flip(), dir_in);
    }

    #[test]
    fn test_edge() {
        let j1 = Spin::new(2).unwrap();
        let edge = Edge::incoming(j1);

        assert_eq!(edge.j, j1);
        assert_eq!(edge.dir, Direction::Incoming);
        assert_eq!(edge.dimension(), 3);

        let flipped = edge.with_flipped_direction();
        assert_eq!(flipped.dir, Direction::Outgoing);
    }

    #[test]
    fn test_cg_spec() {
        let j1 = Spin::new(2).unwrap();
        let j2 = Spin::new(2).unwrap();
        let j3 = Spin::new(2).unwrap();

        let edges = vec![
            Edge::incoming(j1),
            Edge::incoming(j2),
            Edge::outgoing(j3),
        ];

        // For 3 edges, need 1 internal spin per alpha
        let j12 = Spin::new(2).unwrap();
        let alphas = vec![vec![j12]];
        let cg = CGSpec::new(edges, alphas).unwrap();

        assert_eq!(cg.num_external(), 3);
        assert_eq!(cg.om_dimension(), 1);
        assert_eq!(cg.edges.len(), 3);
        assert_eq!(cg.edge_spin_at(0).unwrap(), j1);

        // Test shape computation
        let shape = cg.shape();
        assert_eq!(shape, vec![3, 3, 3, 1]); // [dim(j1), dim(j2), dim(j3), om_dim]
    }

    #[test]
    fn test_contraction() {
        let contr = Contraction::empty();
        assert!(contr.is_empty());
        assert_eq!(contr.num_pairs(), 0);

        // Single contraction
        let contr1 = Contraction::new(&[1], &[0]);
        assert_eq!(contr1.num_pairs(), 1);
        assert!(!contr1.is_empty());

        // Multiple contractions (tensordot style)
        let contr2 = Contraction::new(&[0, 2], &[1, 3]);
        assert_eq!(contr2.num_pairs(), 2);
        assert_eq!(contr2.axes_a, vec![0, 2]);
        assert_eq!(contr2.axes_b, vec![1, 3]);
    }
}
