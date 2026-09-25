"""Compatibility shim re-exporting the moved offline bundle runtime."""

from benchmark_runner.runtime import bundle as _runtime
from benchmark_runner.runtime.bundle import *  # noqa: F403


# A star import skips private names; these are the ones callers still use.
_config = _runtime._config
_filters = _runtime._filters
_hash = _runtime._hash
_is_helper = _runtime._is_helper
_json = _runtime._json
_provenance = _runtime._provenance
_source_key = _runtime._source_key


if __name__ == "__main__":
    raise SystemExit(_runtime.main())
