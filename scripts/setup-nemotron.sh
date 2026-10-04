#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
  lock=requirements/nemotron-mac.lock
elif [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]; then
  lock=requirements/nemotron-cuda.lock
else
  echo "Nemotron setup supports Apple Silicon macOS and x86_64 Linux." >&2
  exit 1
fi
UV="${VIDEO_ANALYZER_UV:-uv}"
command -v "$UV" >/dev/null || { echo "Run mise install --locked first." >&2; exit 1; }
nemotron_python="$(mise where python@3.12.15)/bin/python"
[[ -x "$nemotron_python" ]] || { echo "Run mise install --locked first." >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export UV_PYTHON_DOWNLOADS=never
"$UV" venv .runtime/nemotron --python "$nemotron_python" --allow-existing
"$UV" pip sync "$lock" --python .runtime/nemotron/bin/python --require-hashes
echo "Installed Nemotron. Prepare models with --diarization-backend nemotron."
