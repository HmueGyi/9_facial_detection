#!/usr/bin/env bash
# Create or update the Conda env from environment.yml (project root).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda &>/dev/null; then
  echo "Error: conda not found. Install Miniforge or Miniconda: https://github.com/conda-forge/miniforge/releases"
  exit 1
fi

echo "Project root: $(pwd)"

if conda run -n facial-emotion python -c "pass" 2>/dev/null; then
  echo "Updating existing env: facial-emotion"
  conda env update -f environment.yml --prune
else
  echo "Creating env: facial-emotion"
  conda env create -f environment.yml
fi

echo ""
echo "Next: conda activate facial-emotion"
