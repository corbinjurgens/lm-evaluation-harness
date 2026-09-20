#!/bin/bash

set -euo pipefail

HARNESS="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)"
PYTHON="$HARNESS/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Python 3.9 or newer is required." >&2
        exit 1
    fi
    echo "Creating $HARNESS/.venv..."
    python3 -m venv "$HARNESS/.venv"
fi

# On macOS, use the optional Keychain entry when it exists. Otherwise the
# shared runner securely prompts; pressing Enter selects a local placeholder.
if [[ -z "${OPENAI_API_KEY:-}" ]] && command -v security >/dev/null 2>&1; then
    if KEYCHAIN_OPENAI_KEY="$(security find-generic-password -a "$USER" -s "lm-eval-openai-api-key" -w 2>/dev/null)"; then
        export OPENAI_API_KEY="$KEYCHAIN_OPENAI_KEY"
        unset KEYCHAIN_OPENAI_KEY
    fi
fi

if [[ -z "${HF_TOKEN:-}" ]] && command -v security >/dev/null 2>&1; then
    if KEYCHAIN_HF_TOKEN="$(security find-generic-password -a "$USER" -s "lm-eval-hf-token" -w 2>/dev/null)"; then
        export HF_TOKEN="$KEYCHAIN_HF_TOKEN"
        unset KEYCHAIN_HF_TOKEN
    fi
fi

exec "$PYTHON" "$HARNESS/scripts/run_all_benchmarks.py" "$@"
