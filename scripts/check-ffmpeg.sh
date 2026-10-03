#!/usr/bin/env bash
set -euo pipefail
for executable in ffmpeg ffprobe; do
  if ! command -v "$executable" >/dev/null; then
    echo "Missing $executable. Install FFmpeg first: brew install ffmpeg (Mac) or sudo apt-get install ffmpeg (Debian/Ubuntu)." >&2
    exit 1
  fi
done
