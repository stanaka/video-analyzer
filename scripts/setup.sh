#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) exec bash scripts/setup-mac.sh ;;
  Linux-x86_64) exec bash scripts/setup-cuda.sh ;;
  *)
    echo "Supported platforms: Apple Silicon macOS or x86_64 Linux with NVIDIA CUDA." >&2
    exit 1
    ;;
esac
