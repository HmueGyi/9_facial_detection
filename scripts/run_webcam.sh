#!/usr/bin/env bash
# Run inference from repo root. Uses active facial-emotion env or conda run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" == "facial-emotion" ]]; then
  exec python src/webcam_detect.py "$@"
fi
if command -v conda &>/dev/null; then
  exec conda run -n facial-emotion --no-capture-output python src/webcam_detect.py "$@"
fi
exec python src/webcam_detect.py "$@"
