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

//! Atomic tensor building blocks
//!
//! Functions to construct basic tensor elements: CG3 vertices and connectors.

use crate::core::{Direction, Spin};
use crate::error::Result;
use crate::primitives::{clebsch_gordan, g};
use ndarray::{Array, ArrayD, IxDyn};

/// Build a CG3 (Clebsch-Gordan 3-j) tensor array
///
/// Represents the CG coefficient structure: ⟨j_a, m_a; j_b, m_b | j_c, m_c⟩
/// with canonical arrow directions: legs a and b are incoming, leg c is outgoing.
///
/// # Arguments
/// * `ja` - First input spin (incoming)
/// * `jb` - Second input spin (incoming)
/// * `jc` - Output spin (outgoing)
///
/// # Returns
/// 3-index array V[i_a, i_b, i_c] with CG coefficients
pub fn build_cg3(
    ja: Spin,
    jb: Spin,
    jc: Spin,
) -> Result<ArrayD<f64>> {
    let dim_a = ja.dimension();
    let dim_b = jb.dimension();
    let dim_c = jc.dimension();

    let mut data = Array::zeros(IxDyn(&[dim_a, dim_b, dim_c]));

    // Iterate over all magnetic numbers
    for ia in 0..dim_a {
        for ib in 0..dim_b {
            for ic in 0..dim_c {
                let ma = crate::core::MagneticNumber::from_index(ia, ja);
                let mb = crate::core::MagneticNumber::from_index(ib, jb);
                let mc = crate::core::MagneticNumber::from_index(ic, jc);

                let cg = clebsch_gordan(ja, ma, jb, mb, jc, mc);
                data[[ia, ib, ic]] = cg;
            }
        }
    }

    Ok(data)
}

/// Build a connector array for contracting two legs
///
/// When contracting legs with the same spin:
/// - Opposite directions: identity δ_{m_x, m_y}
/// - Same directions: metric g_{m_x, m_y}
///
/// # Arguments
/// * `j` - Spin quantum number
/// * `dir_x` - Direction of first leg
/// * `dir_y` - Direction of second leg
///
/// # Returns
/// 2-index array C[i_x, i_y]
pub fn build_connector(
    j: Spin,
    dir_x: Direction,
    dir_y: Direction,
) -> Result<ArrayD<f64>> {
    let dim = j.dimension();
    let mut data = Array::zeros(IxDyn(&[dim, dim]));

    for ix in 0..dim {
        for iy in 0..dim {
            let mx = crate::core::MagneticNumber::from_index(ix, j);
            let my = crate::core::MagneticNumber::from_index(iy, j);

            let value = if dir_x != dir_y {
                // Opposite directions: identity
                if mx.twice() == my.twice() {
                    1.0
                } else {
                    0.0
                }
            } else {
                // Same directions: metric
                g(j, mx, my)
            };

            data[[ix, iy]] = value;
        }
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Direction;
    use approx::assert_relative_eq;

    #[test]
    fn test_build_cg3() {
        let j_half = Spin::new(1).unwrap();
        let j_singlet = Spin::new(0).unwrap();

        let vertex = build_cg3(
            j_half,
            j_half,
            j_singlet,
        )
        .unwrap();

        assert_eq!(vertex.ndim(), 3);
        assert_eq!(vertex.shape(), &[2, 2, 1]);

        // Check singlet coupling: CG coefficients for j=1/2 ⊗ j=1/2 → j=0
        let v_up_down = vertex[[0, 1, 0]];
        let v_down_up = vertex[[1, 0, 0]];

        // Both should have magnitude 1/√2
        assert_relative_eq!(v_up_down.abs(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(v_down_up.abs(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);

        // They should be antisymmetric (singlet)
        assert_relative_eq!(v_up_down, -v_down_up, epsilon = 1e-10);
    }

    #[test]
    fn test_build_connector_identity() {
        let j1 = Spin::new(2).unwrap();

        // Opposite directions -> identity
        let conn = build_connector(
            j1,
            Direction::Incoming,
            Direction::Outgoing,
        )
        .unwrap();

        assert_eq!(conn.ndim(), 2);
        assert_eq!(conn.shape(), &[3, 3]);

        // Should be identity matrix
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(conn[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_build_connector_metric() {
        let j1 = Spin::new(2).unwrap();

        // Same directions -> metric
        let conn = build_connector(
            j1,
            Direction::Incoming,
            Direction::Incoming,
        )
        .unwrap();

        assert_eq!(conn.ndim(), 2);

        // Check anti-diagonal elements
        assert_relative_eq!(conn[[0, 2]], 1.0, epsilon = 1e-10); // m=-1 to m=1
        assert_relative_eq!(conn[[1, 1]], -1.0, epsilon = 1e-10); // m=0 to m=0
        assert_relative_eq!(conn[[2, 0]], 1.0, epsilon = 1e-10); // m=1 to m=-1
    }
}
