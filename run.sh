#!/usr/bin/env bash
# Launch Drawtex using Windows-native Python from WSL
# This gives proper tablet/pen input via the Windows display stack

VENV_PYTHON="/mnt/c/Users/oppen/.venvs/drawtex/Scripts/python.exe"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Windows venv not found at $VENV_PYTHON"
    echo "Run setup_windows.bat from a Windows terminal first."
    exit 1
fi

cd "$PROJECT_DIR"
"$VENV_PYTHON" core/main.py "$@"
