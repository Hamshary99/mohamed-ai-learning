@echo off
setlocal

REM Auto-detect virtualenv Python across project directories
if exist "%~dp0..\..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\..\.venv\Scripts\python.exe"
) else if exist "%~dp0..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\.venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

cd /d "%~dp0"
echo ============================================================
echo   StudentGator AI Advisory - Ingesting Document Registry
echo   Python: %PYTHON_EXE%
echo ============================================================

"%PYTHON_EXE%" -m src.ingestion
if errorlevel 1 (
    echo.
    echo [ERROR] Ingestion failed.
    pause
) else (
    echo.
    echo Ingestion completed successfully.
    pause
)

