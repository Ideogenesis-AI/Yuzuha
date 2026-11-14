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

//! SQLite cache for canonical basis data
//!
//! This module provides persistent caching of expensive canonical basis
//! computations using an SQLite database. The cache stores basis arrays
//! in NumPy .npy format, keyed by the array of spin quantum numbers.

use crate::core::Spin;
use crate::error::{Result, YuzuhaError};
use ndarray::ArrayD;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use once_cell::sync::Lazy;
use rusqlite::{Connection, params};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Mutex;

/// Global database connection with lazy initialization
/// Stores (path, connection) to detect when path changes
static DB_CONNECTION: Lazy<Mutex<Option<(PathBuf, Connection)>>> = Lazy::new(|| Mutex::new(None));

// Override cache path per-thread for test isolation (unit and integration tests).
// Using thread-local storage ensures parallel tests do not interfere with each other.
thread_local! {
    static TEST_CACHE_PATH: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
}

/// Apply the path precedence rules: thread-local override → env var → default
///
/// Kept separate from `get_db_path` so the precedence can be tested without
/// mutating the process-global environment, which would race with the other
/// tests running in parallel.
fn resolve_db_path(thread_override: Option<PathBuf>, env_dir: Option<String>) -> PathBuf {
    // Thread-local override takes priority (set by TestCacheGuard in both unit and integration tests)
    if let Some(path) = thread_override {
        return path;
    }

    // YUZUHA_CACHE_PATH should be a directory, we append the filename
    match env_dir {
        Some(dir_path) => PathBuf::from(dir_path).join("cgbasis.db"),
        None => PathBuf::from(".yuzuha/cgbasis.db"),
    }
}

/// Get the database path: thread-local override → env var → default
fn get_db_path() -> PathBuf {
    resolve_db_path(
        TEST_CACHE_PATH.with(|p| p.borrow().clone()),
        std::env::var("YUZUHA_CACHE_PATH").ok(),
    )
}

/// Initialize the database connection and schema
fn init_db(db_path: PathBuf) -> Result<(PathBuf, Connection)> {
    // Create directory if it doesn't exist
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| YuzuhaError::CacheError(format!("Failed to create cache directory: {}", e)))?;
    }
    
    let conn = Connection::open(&db_path)
        .map_err(|e| YuzuhaError::CacheError(format!("Failed to open database: {}", e)))?;
    
    // Create table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cgbasis (
            spins BLOB PRIMARY KEY,
            data BLOB NOT NULL
        )",
        [],
    )
    .map_err(|e| YuzuhaError::CacheError(format!("Failed to create table: {}", e)))?;
    
    Ok((db_path, conn))
}

/// Get or initialize the database connection
/// 
/// Automatically reconnects if the cache path has changed (important for test isolation)
fn get_connection() -> Result<std::sync::MutexGuard<'static, Option<(PathBuf, Connection)>>> {
    let mut guard = DB_CONNECTION.lock()
        .map_err(|e| YuzuhaError::CacheError(format!("Failed to acquire lock: {}", e)))?;
    
    let current_path = get_db_path();
    
    // Initialize if no connection, or reconnect if path changed
    let needs_init = match guard.as_ref() {
        None => true,
        Some((stored_path, _)) => stored_path != &current_path,
    };
    
    if needs_init {
        *guard = Some(init_db(current_path)?);
    }
    
    Ok(guard)
}

/// Serialize spin array to bytes for use as database key
fn serialize_spins(spins: &[Spin]) -> Vec<u8> {
    spins.iter()
        .flat_map(|s| s.twice().to_le_bytes())
        .collect()
}

/// Serialize ArrayD to .npy format bytes
fn serialize_array(array: &ArrayD<f64>) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    // write_npy takes W: Write by value; &mut Vec<u8> implements Write
    array.write_npy(&mut buf)
        .map_err(|e| YuzuhaError::CacheError(format!("Failed to serialize array: {}", e)))?;
    Ok(buf)
}

/// Deserialize .npy format bytes to ArrayD
fn deserialize_array(bytes: &[u8]) -> Result<ArrayD<f64>> {
    // read_npy takes R: Read by value; Cursor<&[u8]> implements Read
    let reader = Cursor::new(bytes);
    ArrayD::<f64>::read_npy(reader)
        .map_err(|e| YuzuhaError::CacheError(format!("Failed to deserialize array: {}", e)))
}

/// Query cached canonical basis data
///
/// # Arguments
/// * `spins` - Array of spin quantum numbers
///
/// # Returns
/// Some(ArrayD) if cached data exists, None otherwise
pub fn query_canonical_basis(spins: &[Spin]) -> Result<Option<ArrayD<f64>>> {
    let mut conn_guard = get_connection()?;
    let (_, conn) = conn_guard.as_mut()
        .ok_or_else(|| YuzuhaError::CacheError("Database connection not initialized".to_string()))?;
    
    let key = serialize_spins(spins);
    
    let mut stmt = conn.prepare("SELECT data FROM cgbasis WHERE spins = ?")
        .map_err(|e| YuzuhaError::CacheError(format!("Failed to prepare query: {}", e)))?;
    
    let result = stmt.query_row(params![key], |row| {
        let bytes: Vec<u8> = row.get(0)?;
        Ok(bytes)
    });
    
    match result {
        Ok(bytes) => {
            let array = deserialize_array(&bytes)?;
            Ok(Some(array))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(YuzuhaError::CacheError(format!("Query failed: {}", e))),
    }
}

/// Store canonical basis data in cache
///
/// # Arguments
/// * `spins` - Array of spin quantum numbers
/// * `data` - Canonical basis array to cache
pub fn store_canonical_basis(spins: &[Spin], data: &ArrayD<f64>) -> Result<()> {
    let mut conn_guard = get_connection()?;
    let (_, conn) = conn_guard.as_mut()
        .ok_or_else(|| YuzuhaError::CacheError("Database connection not initialized".to_string()))?;
    
    let key = serialize_spins(spins);
    let value = serialize_array(data)?;
    
    conn.execute(
        "INSERT OR REPLACE INTO cgbasis (spins, data) VALUES (?1, ?2)",
        params![key, value],
    )
    .map_err(|e| YuzuhaError::CacheError(format!("Failed to insert data: {}", e)))?;
    
    Ok(())
}

/// Reset the database connection (used for testing)
#[cfg(any(test, feature = "test-utils"))]
fn reset_connection() -> Result<()> {
    let mut guard = DB_CONNECTION.lock()
        .map_err(|e| YuzuhaError::CacheError(format!("Failed to acquire lock: {}", e)))?;
    *guard = None;
    Ok(())
}

/// RAII guard for test cache isolation.
///
/// Sets up a temporary cache directory and ensures cleanup on drop.
/// This prevents tests from reading from or polluting the production cache.
///
/// Only available with the `test-utils` feature (or inside `#[cfg(test)]` contexts).
/// Enable for integration tests with: `cargo test --features test-utils`
///
/// Each instance creates a unique temporary cache directory using thread-local
/// storage, ensuring parallel tests do not interfere with each other.
///
/// # Example
/// ```no_run
/// use yuzuha::builders::TestCacheGuard;
///
/// #[test]
/// fn my_test() {
///     let _guard = TestCacheGuard::new();
///     // ... test code that uses build_canonical_basis_data ...
///     // Cache is automatically cleaned up when _guard is dropped
/// }
/// ```
#[cfg(any(test, feature = "test-utils"))]
pub struct TestCacheGuard {
    temp_dir: std::path::PathBuf,
}

/// Token that names a temporary directory no other guard can be using.
///
/// The process id and the clock separate concurrent runs, but not tests within
/// one run: the wall clock advances in microsecond steps on some platforms, so
/// two guards created in the same instant would otherwise share a directory
/// and the first to drop would delete the database the other is still writing.
/// The counter makes the name unique within the process.
#[cfg(any(test, feature = "test-utils"))]
fn unique_suffix() -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    format!(
        "{}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

#[cfg(any(test, feature = "test-utils"))]
impl TestCacheGuard {
    /// Create a new test cache in a unique temporary directory.
    pub fn new() -> Self {
        use std::fs;

        let temp_dir = std::env::temp_dir().join(format!("yuzuha_test_cache_{}", unique_suffix()));

        fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

        TEST_CACHE_PATH.with(|p| {
            *p.borrow_mut() = Some(temp_dir.join("cgbasis.db"));
        });

        reset_connection().expect("Failed to reset connection");
        
        TestCacheGuard { temp_dir }
    }

    /// Path of the database this guard redirects the cache to.
    pub fn db_path(&self) -> PathBuf {
        self.temp_dir.join("cgbasis.db")
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl Drop for TestCacheGuard {
    fn drop(&mut self) {
        use std::fs;

        TEST_CACHE_PATH.with(|p| {
            *p.borrow_mut() = None;
        });
        let _ = reset_connection();
        let _ = fs::remove_dir_all(&self.temp_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_serialize_deserialize_array() {
        let _guard = TestCacheGuard::new();
        
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
        let bytes = serialize_array(&array).unwrap();
        let recovered = deserialize_array(&bytes).unwrap();
        
        assert_eq!(array.shape(), recovered.shape());
        assert_eq!(array.as_slice(), recovered.as_slice());
    }

    #[test]
    fn test_serialize_spins() {
        let _guard = TestCacheGuard::new();
        
        let j_half = Spin::new(1).unwrap();
        let j1 = Spin::new(2).unwrap();
        let spins = vec![j_half, j1, j_half];
        
        let bytes = serialize_spins(&spins);
        
        // Should be 3 * 4 bytes (3 i32 values)
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn test_cache_store_and_query() {
        let _guard = TestCacheGuard::new();
        
        let j1 = Spin::new(2).unwrap();
        let spins = vec![j1, j1, j1];
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
        
        // Store data
        store_canonical_basis(&spins, &array).unwrap();
        
        // Query should return the stored data
        let retrieved = query_canonical_basis(&spins).unwrap();
        assert!(retrieved.is_some());
        
        let retrieved_array = retrieved.unwrap();
        assert_eq!(array.shape(), retrieved_array.shape());
        assert_eq!(array.as_slice(), retrieved_array.as_slice());
    }

    #[test]
    fn test_cache_miss() {
        let _guard = TestCacheGuard::new();
        
        let j1 = Spin::new(2).unwrap();
        let spins = vec![j1, j1, j1];
        
        // Query non-existent data should return None
        let result = query_canonical_basis(&spins).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_db_path_precedence() {
        let thread_path = PathBuf::from("/tmp/thread-local/cgbasis.db");

        // The thread-local override wins even when the env var is set.
        assert_eq!(
            resolve_db_path(Some(thread_path.clone()), Some("/tmp/from-env".to_string())),
            thread_path
        );
        assert_eq!(resolve_db_path(Some(thread_path.clone()), None), thread_path);

        // The env var names a directory; the file name is appended.
        assert_eq!(
            resolve_db_path(None, Some("/tmp/from-env".to_string())),
            PathBuf::from("/tmp/from-env/cgbasis.db")
        );

        // With neither override, the cache lives beside the working directory.
        assert_eq!(resolve_db_path(None, None), PathBuf::from(".yuzuha/cgbasis.db"));
    }

    #[test]
    fn test_get_db_path_uses_thread_local_override() {
        let guard = TestCacheGuard::new();
        assert_eq!(get_db_path(), guard.db_path());
    }

    #[test]
    fn test_init_db_creates_missing_parent_directory() {
        let root = std::env::temp_dir().join(format!("yuzuha_init_db_{}", unique_suffix()));
        let db_path = root.join("nested").join("cgbasis.db");
        assert!(!db_path.parent().unwrap().exists());

        let (stored_path, conn) = init_db(db_path.clone()).unwrap();
        assert_eq!(stored_path, db_path);
        assert!(db_path.parent().unwrap().exists());

        // The schema must be usable immediately after initialization.
        conn.execute("INSERT INTO cgbasis (spins, data) VALUES (x'00', x'01')", [])
            .unwrap();

        drop(conn);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_init_db_reports_unusable_parent_directory() {
        // A regular file cannot act as a parent directory, so create_dir_all fails.
        let blocker = std::env::temp_dir().join(format!("yuzuha_blocker_{}", unique_suffix()));
        std::fs::write(&blocker, b"not a directory").unwrap();

        let err = init_db(blocker.join("cgbasis.db")).unwrap_err();
        match err {
            YuzuhaError::CacheError(msg) => {
                assert!(msg.contains("Failed to create cache directory"), "got: {msg}")
            }
            other => panic!("expected CacheError, got {other:?}"),
        }

        let _ = std::fs::remove_file(&blocker);
    }

    #[test]
    fn test_deserialize_array_rejects_malformed_bytes() {
        let err = deserialize_array(b"this is not a .npy payload").unwrap_err();
        match err {
            YuzuhaError::CacheError(msg) => {
                assert!(msg.contains("Failed to deserialize array"), "got: {msg}")
            }
            other => panic!("expected CacheError, got {other:?}"),
        }
    }

    #[test]
    fn test_query_reports_unreadable_row() {
        let _guard = TestCacheGuard::new();

        let j1 = Spin::new(2).unwrap();
        let spins = vec![j1, j1];
        let key = serialize_spins(&spins);

        // SQLite is dynamically typed, so a non-blob slips past the schema and
        // only fails when the row is read back as bytes.
        {
            let mut conn_guard = get_connection().unwrap();
            let (_, conn) = conn_guard.as_mut().unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO cgbasis (spins, data) VALUES (?1, 42)",
                params![key],
            )
            .unwrap();
        }

        let err = query_canonical_basis(&spins).unwrap_err();
        match err {
            YuzuhaError::CacheError(msg) => assert!(msg.contains("Query failed"), "got: {msg}"),
            other => panic!("expected CacheError, got {other:?}"),
        }
    }

    #[test]
    fn test_connection_reopens_when_path_changes() {
        let first = TestCacheGuard::new();
        let j1 = Spin::new(2).unwrap();
        let spins = vec![j1, j1, j1];
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
        store_canonical_basis(&spins, &array).unwrap();

        {
            let guard = get_connection().unwrap();
            assert_eq!(guard.as_ref().unwrap().0, first.db_path());
        }

        // A second guard repoints the thread-local path. The cached connection
        // must be discarded, so the entry written above is no longer visible.
        let second = TestCacheGuard::new();
        assert_ne!(second.db_path(), first.db_path());

        {
            let guard = get_connection().unwrap();
            assert_eq!(guard.as_ref().unwrap().0, second.db_path());
        }
        assert!(query_canonical_basis(&spins).unwrap().is_none());
    }
}
