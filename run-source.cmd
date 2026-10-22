@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment. Python 3.12 x64 is recommended.
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12 x64 from python.org, then run this file again.
    pause
    exit /b 1
  )
)
rem 2.1.0: install again whenever requirements.txt changed since the last install (e.g. PyAV for
rem time-lapse videos, new in 2.0), not only when an older module is missing.
set "NEED_INSTALL="
fc /b requirements.txt ".venv\lumen-requirements.txt" >nul 2>&1
if errorlevel 1 set "NEED_INSTALL=1"
".venv\Scripts\python.exe" -c "import numpy, cv2, PIL, rawpy, PySide6, tifffile, onnxruntime, av" >nul 2>&1
if errorlevel 1 set "NEED_INSTALL=1"
if defined NEED_INSTALL (
  ".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
  if errorlevel 1 (
    pause
    exit /b 1
  )
  copy /y requirements.txt ".venv\lumen-requirements.txt" >nul
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
