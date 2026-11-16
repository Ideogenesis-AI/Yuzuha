# Database

`startup_database()` pre-computes the canonical CG bases for the spin configurations a
tensor network calculation is most likely to need, so that the cost is paid once at
startup instead of as a cache miss in the middle of a run. It is purely an optimization:
the cache fills itself on first use either way, and the results are identical.

::: yuzuha.startup_database
    options:
      show_source: false
      heading_level: 2

## Description

The default sweep covers the configurations that dominate practical SU(2) tensor network
work:

- **3rd-order bases**: one spin in \(\{1/2, 1\}\), the other two in
  \(\{1/2, 1, 3/2, 2, 5/2, 3\}\) — all 3! orderings of each triplet
- **4th-order bases**: two spins fixed at \(1/2\), two in
  \(\{1/2, 1, 3/2, 2, 5/2, 3\}\) — all \(\binom{4}{2} = 6\) positions for the
  two fixed spins

Configurations already present in the database are not recomputed, so calling the
function repeatedly is cheap and idempotent.

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

# Narrow the sweep — doubled spin values, so 1 is j=1/2 and 2 is j=1
yuzuha.startup_database(special_spins=[1], general_spins=[1, 2])
```

`special_spins` is the grid for the distinguished edge of the 3rd-order configurations and
`general_spins` the grid for the remaining edges, both as doubled values. Omitting them
uses the default ranges above. Narrowing them is worthwhile when an application only ever
sees a few spin values, or in tests where the full sweep would dominate the runtime.

### Progress Logging

Progress is logged at `INFO` level through the `yuzuha.database` logger, which reports:

- the number of configurations to process
- a running count, every 10 configurations
- any configuration skipped for a triangle inequality violation
- a final tally of successes and failures

Pass `log_level=logging.WARNING` to silence all of it, or `logger=` to route it into an
application's own logging setup.

### Database Location

The canonical basis cache is written to `cgbasis.db` in the Yuzuha cache directory,
resolved in the same priority order as the symbol caches:

1. Thread-local test override, for test isolation
2. `YUZUHA_CACHE_PATH` environment variable
3. Default: `.yuzuha/` in the current working directory

## See Also

- [Cache Management](cache.md): Control cache paths and inspect statistics
- [Canonical Basis](../symbols/canonical-basis.md): The function whose results are cached
- [Yuzuha Protocol — Data Caching](../../protocol/caching.md): Formal caching
  specification

## Notes

Triangle inequality violations are not errors. A swept configuration for which no valid
internal coupling exists is logged at `DEBUG` level, omitted from the success count, and
skipped — so the number of entries created is smaller than the size of the grid swept.
