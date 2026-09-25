"""Host entry point for the benchmark runner.

Standard library only: inserts the checkout root onto sys.path so
benchmark_runner is importable without installing the package, then
delegates to its CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_runner.cli import main


raise SystemExit(main())
