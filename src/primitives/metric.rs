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

//! Invariant metric for SU(2) arrow reversal
//!
//! The metric g implements arrow reversal via:
//! g^(j)_{m,m'} = (-1)^(j-m) δ_{m,-m'}
//!
//! This allows converting between incoming and outgoing edges.

use crate::core::{Direction, Edge, MagneticNumber, Spin};

/// Compute the invariant metric element g^(J)_{M,M'}
///
/// Returns (-1)^((J-M)/2) if M' = -M, otherwise 0.
///
/// # Arguments
/// * `j` - Spin quantum number
/// * `m` - First magnetic quantum number
/// * `mp` - Second magnetic quantum number (M')
///
/// # Returns
/// The metric value: (-1)^((J-M)/2) if M' = -M, else 0
pub fn g(j: Spin, m: MagneticNumber, mp: MagneticNumber) -> f64 {
    if mp.twice() != -m.twice() {
        return 0.0;
    }

    let j_val = j.twice();
    let m_val = m.twice();
    let exponent = (j_val - m_val) / 2;

    if exponent % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Use an edge as incoming, applying metric if necessary
///
/// If the edge is already incoming, returns (M, 1.0).
/// If the edge is outgoing, returns (-M, g(J,M,-M)).
///
/// # Arguments
/// * `edge` - The edge to use
/// * `m` - The magnetic quantum number
///
/// # Returns
/// Tuple of (transformed M, phase factor)
pub fn use_as_incoming(edge: &Edge, m: MagneticNumber) -> (MagneticNumber, f64) {
    match edge.dir {
        Direction::Incoming => (m, 1.0),
        Direction::Outgoing => {
            let m_flipped = m.negate();
            let phase = g(edge.j, m, m_flipped);
            (m_flipped, phase)
        }
    }
}

/// Use an edge as outgoing, applying metric if necessary
///
/// If the edge is already outgoing, returns (M, 1.0).
/// If the edge is incoming, returns (-M, g(J,M,-M)).
///
/// # Arguments
/// * `edge` - The edge to use
/// * `m` - The magnetic quantum number
///
/// # Returns
/// Tuple of (transformed M, phase factor)
pub fn use_as_outgoing(edge: &Edge, m: MagneticNumber) -> (MagneticNumber, f64) {
    match edge.dir {
        Direction::Outgoing => (m, 1.0),
        Direction::Incoming => {
            let m_flipped = m.negate();
            let phase = g(edge.j, m, m_flipped);
            (m_flipped, phase)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g_metric() {
        let j1 = Spin::new(2).unwrap(); // j=1

        // g^(1)_{1,-1} = (-1)^((2-2)/2) = (-1)^0 = 1
        let m1 = MagneticNumber::new_unchecked(2);
        let m_neg1 = MagneticNumber::new_unchecked(-2);
        assert_eq!(g(j1, m1, m_neg1), 1.0);

        // g^(1)_{0,0} = (-1)^((2-0)/2) = (-1)^1 = -1
        let m0 = MagneticNumber::new_unchecked(0);
        assert_eq!(g(j1, m0, m0.negate()), -1.0);

        // g^(1)_{-1,1} = (-1)^((2-(-2))/2) = (-1)^2 = 1
        assert_eq!(g(j1, m_neg1, m1), 1.0);

        // g^(1)_{1,1} = 0 (wrong sign)
        assert_eq!(g(j1, m1, m1), 0.0);
    }

    #[test]
    fn test_use_as_incoming() {
        let j1 = Spin::new(2).unwrap();
        let m1 = MagneticNumber::new_unchecked(2);

        // Incoming edge: no change
        let edge_in = Edge::incoming(j1);
        let (m_out, phase) = use_as_incoming(&edge_in, m1);
        assert_eq!(m_out.twice(), 2);
        assert_eq!(phase, 1.0);

        // Outgoing edge: flip with metric
        let edge_out = Edge::outgoing(j1);
        let (m_out, phase) = use_as_incoming(&edge_out, m1);
        assert_eq!(m_out.twice(), -2);
        assert_eq!(phase, g(j1, m1, m_out));
    }

    #[test]
    fn test_use_as_outgoing() {
        let j1 = Spin::new(2).unwrap();
        let m1 = MagneticNumber::new_unchecked(2);

        // Outgoing edge: no change
        let edge_out = Edge::outgoing(j1);
        let (m_out, phase) = use_as_outgoing(&edge_out, m1);
        assert_eq!(m_out.twice(), 2);
        assert_eq!(phase, 1.0);

        // Incoming edge: flip with metric
        let edge_in = Edge::incoming(j1);
        let (m_out, phase) = use_as_outgoing(&edge_in, m1);
        assert_eq!(m_out.twice(), -2);
        assert_eq!(phase, g(j1, m1, m_out));
    }
}
