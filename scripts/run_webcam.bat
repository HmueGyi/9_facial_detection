@echo off
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if "%CONDA_DEFAULT_ENV%"=="facial-emotion" (
  python src\webcam_detect.py %*
  exit /b %ERRORLEVEL%
)
where conda >nul 2>&1
if not errorlevel 1 (
  conda run -n facial-emotion python src\webcam_detect.py %*
  exit /b %ERRORLEVEL%
)
python src\webcam_detect.py %*
