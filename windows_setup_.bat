@echo off
REM Setup Windows Python virtual environment for Drawtex
REM Run this from any Windows terminal (CMD or PowerShell)

setlocal

REM Switch to a local drive to avoid UNC path issues
cd /d C:\

set "VENV_DIR=C:\Users\oppen\.venvs\drawtex"
set "REQUIREMENTS=\\wsl.localhost\Ubuntu\home\oppen\Drawtex\requirements_windows.txt"

echo === Drawtex Windows Environment Setup ===
echo.

REM Check Python
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12 from python.org
    exit /b 1
)

echo [1/3] Creating virtual environment at %VENV_DIR% ...
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo       Venv already exists, skipping creation.
) else (
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        exit /b 1
    )
)

echo [2/3] Upgrading pip ...
call "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo [3/3] Installing dependencies from requirements.txt ...
call "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQUIREMENTS%" --quiet
if errorlevel 1 echo WARNING: Some packages may have failed. Linux-only GPU packages are expected to fail on Windows. This is OK.

echo.
echo === Setup complete ===
echo Venv location: %VENV_DIR%
echo.
echo To run Drawtex from WSL:  ./run.sh
