@echo off
setlocal

REM Auto-detect virtualenv Python across project directories
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else if exist "%~dp0..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\.venv\Scripts\python.exe"
) else if exist "%~dp0..\..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\..\.venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

cd /d "%~dp0"
"%PYTHON_EXE%" test_client.py %*
if errorlevel 1 (
    echo.
    pause
)
