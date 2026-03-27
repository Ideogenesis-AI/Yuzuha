# Database

Pre-population of the canonical basis cache for common spin configurations.

::: yuzuha.startup_database
    options:
      show_source: false
      heading_level: 2

## Description

`startup_database()` pre-computes and caches the canonical CG bases for a standard set
of spin configurations that are commonly encountered in tensor network applications:

- **3rd-order bases**: One spin in \(\{1/2, 1\}\), two spins in
  \(\{1/2, 1, 3/2, 2, 5/2, 3\}\) — all 3! orderings of each triplet
- **4th-order bases**: Two spins fixed at \(1/2\), two spins in
  \(\{1/2, 1, 3/2, 2, 5/2, 3\}\) — all \(\binom{4}{2} = 6\) positions for the
  two fixed spins

Calling this function at application startup avoids cache-miss latency during
the first computation of each configuration.

### Usage

```python
import yuzuha

# Basic usage — pre-populate with default logging
yuzuha.startup_database()

# With a custom logger
import logging
logger = logging.getLogger('my_app')
yuzuha.startup_database(logger=logger)

# Suppress output
yuzuha.startup_database(log_level=logging.WARNING)
```

### Progress Logging

By default, `startup_database` logs progress at `INFO` level through the
`yuzuha.database` logger. Log messages include:

- Number of configurations to process
- Current progress (every 10 configs)
- Configurations skipped due to triangle inequality violations
- Final count of successes and failures

### When to Call

`startup_database()` is **optional** — the cache is populated automatically on first
use. Call it explicitly if you want:

- Predictable latency from the start of a computation
- Logged progress for long-running pre-computation

## Database Location

The canonical basis cache is managed by the Rust layer and stored in a SQLite database
alongside the other Yuzuha caches. The location is controlled by:

1. Thread-local test override (for test isolation)
2. `YUZUHA_CACHE_PATH` environment variable
3. Default: `.yuzuha/` in the current working directory

## See Also

- [Cache Management](cache.md): Control cache paths and inspect statistics
- [Canonical Basis](../symbols/canonical-basis.md): The function whose results are cached
- [Yuzuha Protocol — Data Caching](../../protocol/caching.md): Formal caching
  specification

## Notes

Triangle inequality violations are silently skipped — configurations where no valid
internal spin coupling exists (e.g. three spins where no triplet satisfies the triangle
inequality) are logged at `DEBUG` level and omitted from the count.

Configurations already in the database are not recomputed; subsequent calls to
`startup_database` are idempotent.
