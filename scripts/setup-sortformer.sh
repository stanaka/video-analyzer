#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
  lock=requirements/sortformer-mac.lock
elif [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]; then
  lock=requirements/sortformer-cuda.lock
else
  echo "Sortformer setup supports Apple Silicon macOS and x86_64 Linux." >&2
  exit 1
fi
UV="${VIDEO_ANALYZER_UV:-uv}"
command -v "$UV" >/dev/null || { echo "Run mise install --locked first." >&2; exit 1; }
sortformer_python="$(mise where python@3.12.15)/bin/python"
[[ -x "$sortformer_python" ]] || { echo "Run mise install --locked first." >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export UV_PYTHON_DOWNLOADS=never
"$UV" venv .runtime/sortformer --python "$sortformer_python" --allow-existing
"$UV" pip sync "$lock" --python .runtime/sortformer/bin/python --require-hashes
echo "Installed Sortformer. Prepare models with --diarization-backend sortformer."
