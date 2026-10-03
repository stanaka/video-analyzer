#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "The CUDA lock targets x86_64 Linux." >&2
  exit 1
fi
bash scripts/check-ffmpeg.sh
UV="${VIDEO_ANALYZER_UV:-uv}"
command -v "$UV" >/dev/null || { echo "Run mise install, then mise run setup." >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
"$UV" sync --frozen --python "${UV_PYTHON:-$(cat .python-version)}"
"$UV" venv .runtime/torch --python .venv/bin/python --allow-existing
"$UV" pip sync requirements/torch-cuda.lock --python .runtime/torch/bin/python --require-hashes
echo "Installed. Next: mise run prepare-models --profile cuda --asr both"
echo "Analyze with: mise run analyze input.mp4"
