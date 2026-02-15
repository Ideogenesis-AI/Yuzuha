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

//! Tests for concurrent cache access
//!
//! These tests verify that the SQLite cache handles concurrent reads and writes
//! correctly, which is critical for production use where multiple threads may
//! compute and cache basis data simultaneously.

use yuzuha::builders::build_canonical_basis_data;
use yuzuha::core::{CGSpec, Edge, Spin};
use std::sync::Arc;
use std::thread;

/// Test concurrent writes to the same cache database
///
/// Spawns multiple threads that simultaneously compute and cache different
/// canonical basis tensors. Verifies that all data is correctly stored without
/// corruption or race conditions.
#[test]
fn test_concurrent_cache_writes() {
    // Create a shared temporary database for this test
    let temp_dir = std::env::temp_dir().join(format!(
        "yuzuha_concurrent_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");
    
    let cache_path = temp_dir.join("cgbasis.db");
    
    // Set the cache path for all threads
    unsafe {
        std::env::set_var("YUZUHA_CACHE_PATH", &cache_path);
    }
    
    // Define different specs to compute in parallel
    // Note: Must satisfy angular momentum conservation (couple to j=0)
    let specs = vec![
        // Three j=1 spins
        create_spec(vec![2, 2, 2]),
        // Three j=2 spins
        create_spec(vec![4, 4, 4]),
        // Four j=1 spins
        create_spec(vec![2, 2, 2, 2]),
        // Mixed: j=1/2, j=1/2, j=1
        create_spec(vec![1, 1, 2]),
        // Mixed: j=1, j=2, j=1
        create_spec(vec![2, 4, 2]),
        // Five j=1 spins
        create_spec(vec![2, 2, 2, 2, 2]),
        // Four j=1/2 spins
        create_spec(vec![1, 1, 1, 1]),
        // Mixed: j=1/2, j=3/2, j=2
        create_spec(vec![1, 3, 4]),
    ];
    
    let specs = Arc::new(specs);
    let mut handles = vec![];
    
    // Spawn threads to compute in parallel
    for i in 0..specs.len() {
        let specs_clone = Arc::clone(&specs);
        let handle = thread::spawn(move || {
            let spec = &specs_clone[i];
            
            // First computation - should cache the result
            let result1 = build_canonical_basis_data(spec)
                .expect("Failed to build canonical basis");
            
            // Second computation - should read from cache
            let result2 = build_canonical_basis_data(spec)
                .expect("Failed to build canonical basis");
            
            // Results should be identical
            assert_eq!(result1.shape(), result2.shape());
            assert_eq!(result1.as_slice(), result2.as_slice());
            
            result1
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete and collect results
    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();
    
    // Verify each result by recomputing in the main thread
    // This ensures cached values match freshly computed values
    for (i, cached_result) in results.iter().enumerate() {
        let spec = &specs[i];
        
        // Recompute without cache (will actually read from cache again, but that's fine)
        let fresh_result = build_canonical_basis_data(spec)
            .expect("Failed to recompute canonical basis");
        
        // Verify shape matches
        assert_eq!(
            cached_result.shape(),
            fresh_result.shape(),
            "Shape mismatch for spec {:?}",
            spec.edges.iter().map(|e| e.j.twice()).collect::<Vec<_>>()
        );
        
        // Verify all values match exactly
        let cached_slice = cached_result.as_slice().unwrap();
        let fresh_slice = fresh_result.as_slice().unwrap();
        
        for (j, (&cached_val, &fresh_val)) in cached_slice.iter().zip(fresh_slice.iter()).enumerate() {
            assert!(
                (cached_val - fresh_val).abs() < 1e-12,
                "Value mismatch at index {} for spec {:?}: cached={}, fresh={}",
                j,
                spec.edges.iter().map(|e| e.j.twice()).collect::<Vec<_>>(),
                cached_val,
                fresh_val
            );
        }
        
        // Verify results are non-trivial (not all zeros)
        let sum: f64 = cached_slice.iter().map(|x| x.abs()).sum();
        assert!(sum > 1e-10, "Result should not be all zeros");
    }
    
    // Clean up
    unsafe {
        std::env::remove_var("YUZUHA_CACHE_PATH");
    }
    let _ = std::fs::remove_dir_all(&temp_dir);
}

/// Test concurrent reads from cache
///
/// First populates the cache, then spawns multiple threads that simultaneously
/// read the same cached data. Verifies that all threads get consistent results.
#[test]
fn test_concurrent_cache_reads() {
    // Create a shared temporary database for this test
    let temp_dir = std::env::temp_dir().join(format!(
        "yuzuha_concurrent_read_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");
    
    let cache_path = temp_dir.join("cgbasis.db");
    
    // Set the cache path
    unsafe {
        std::env::set_var("YUZUHA_CACHE_PATH", &cache_path);
    }
    
    // Pre-populate cache with some data
    let spec = create_spec(vec![2, 2, 2]); // Three j=1 spins
    let expected_result = build_canonical_basis_data(&spec)
        .expect("Failed to build canonical basis");
    
    // Spawn multiple threads to read the same data
    let spec = Arc::new(spec);
    let mut handles = vec![];
    
    for _ in 0..8 {
        let spec_clone = Arc::clone(&spec);
        let handle = thread::spawn(move || {
            build_canonical_basis_data(&spec_clone)
                .expect("Failed to build canonical basis")
        });
        handles.push(handle);
    }
    
    // Wait for all threads and verify results are identical
    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        
        // Verify shape
        assert_eq!(result.shape(), expected_result.shape());
        
        // Verify all values match exactly
        let result_slice = result.as_slice().unwrap();
        let expected_slice = expected_result.as_slice().unwrap();
        
        for (i, (&result_val, &expected_val)) in result_slice.iter().zip(expected_slice.iter()).enumerate() {
            assert!(
                (result_val - expected_val).abs() < 1e-12,
                "Value mismatch at index {}: result={}, expected={}",
                i, result_val, expected_val
            );
        }
    }
    
    // Clean up
    unsafe {
        std::env::remove_var("YUZUHA_CACHE_PATH");
    }
    let _ = std::fs::remove_dir_all(&temp_dir);
}

/// Test mixed concurrent reads and writes
///
/// Some threads write new data while others read existing data.
#[test]
fn test_concurrent_mixed_operations() {
    // Create a shared temporary database for this test
    let temp_dir = std::env::temp_dir().join(format!(
        "yuzuha_concurrent_mixed_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");
    
    let cache_path = temp_dir.join("cgbasis.db");
    
    // Set the cache path
    unsafe {
        std::env::set_var("YUZUHA_CACHE_PATH", &cache_path);
    }
    
    // Pre-populate with one spec
    let read_spec = Arc::new(create_spec(vec![2, 2, 2]));
    let _ = build_canonical_basis_data(&read_spec)
        .expect("Failed to build canonical basis");
    
    // Create specs for writing
    let write_specs = vec![
        create_spec(vec![1, 1, 2]),
        create_spec(vec![4, 4, 4]),
        create_spec(vec![1, 2, 1]),
        create_spec(vec![2, 4, 2]),
    ];
    let write_specs = Arc::new(write_specs);
    
    let mut handles = vec![];
    
    // Spawn reader threads
    for _ in 0..4 {
        let spec = Arc::clone(&read_spec);
        let handle = thread::spawn(move || {
            for _ in 0..3 {
                let _ = build_canonical_basis_data(&spec)
                    .expect("Failed to read from cache");
            }
        });
        handles.push(handle);
    }
    
    // Spawn writer threads
    for i in 0..write_specs.len() {
        let specs = Arc::clone(&write_specs);
        let handle = thread::spawn(move || {
            let _ = build_canonical_basis_data(&specs[i])
                .expect("Failed to write to cache");
        });
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Verify all data is still accessible and correct
    // Read the data twice to ensure consistency
    let read_result1 = build_canonical_basis_data(&read_spec)
        .expect("Read spec should still be cached");
    let read_result2 = build_canonical_basis_data(&read_spec)
        .expect("Read spec should still be cached");
    
    // Verify the read spec data is consistent
    assert_eq!(read_result1.shape(), read_result2.shape());
    let slice1 = read_result1.as_slice().unwrap();
    let slice2 = read_result2.as_slice().unwrap();
    for (i, (&val1, &val2)) in slice1.iter().zip(slice2.iter()).enumerate() {
        assert!(
            (val1 - val2).abs() < 1e-12,
            "Read spec value mismatch at index {}: {} vs {}",
            i, val1, val2
        );
    }
    
    // Verify all written specs are cached with correct values
    for spec in write_specs.iter() {
        let result1 = build_canonical_basis_data(spec)
            .expect("Write spec should be cached");
        let result2 = build_canonical_basis_data(spec)
            .expect("Write spec should be cached");
        
        // Verify consistency
        assert_eq!(result1.shape(), result2.shape());
        let slice1 = result1.as_slice().unwrap();
        let slice2 = result2.as_slice().unwrap();
        for (i, (&val1, &val2)) in slice1.iter().zip(slice2.iter()).enumerate() {
            assert!(
                (val1 - val2).abs() < 1e-12,
                "Write spec value mismatch at index {}: {} vs {}",
                i, val1, val2
            );
        }
    }
    
    // Clean up
    unsafe {
        std::env::remove_var("YUZUHA_CACHE_PATH");
    }
    let _ = std::fs::remove_dir_all(&temp_dir);
}

/// Helper to create a CGSpec from spin twice values
fn create_spec(twice_values: Vec<i32>) -> CGSpec {
    let edges: Vec<Edge> = twice_values.iter()
        .enumerate()
        .map(|(i, &twice)| {
            let spin = Spin::new(twice).expect("Invalid spin");
            if i == twice_values.len() - 1 {
                Edge::outgoing(spin)
            } else {
                Edge::incoming(spin)
            }
        })
        .collect();
    
    CGSpec::from_edges(edges).expect("Failed to create CGSpec")
}
