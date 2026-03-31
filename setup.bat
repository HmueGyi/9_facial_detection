@echo off
setlocal
cd /d "%~dp0"

where conda >nul 2>&1
if errorlevel 1 (
  echo Error: conda not found. Install Miniforge or Miniconda.
  echo https://github.com/conda-forge/miniforge/releases
  exit /b 1
)

echo Project root: %CD%

conda run -n facial-emotion python -c "pass" 2>nul
if errorlevel 1 (
  echo Creating env: facial-emotion
  conda env create -f environment.yml
) else (
  echo Updating env: facial-emotion
  conda env update -f environment.yml --prune
)

echo.
echo Next: conda activate facial-emotion
