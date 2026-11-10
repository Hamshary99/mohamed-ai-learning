@echo off
setlocal enabledelayedexpansion

title StudentGator RAG Benchmark - Naive vs. Proper RAG

echo ==============================================================================
echo  STUDENTGATOR AI ENGINEERING: NAIVE vs. PROPER RAG BENCHMARK
echo ==============================================================================
echo.

cd /d "%~dp0"

set "PYTHON_EXE="

if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0..\..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0..\..\..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\..\..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python environment was not found!
    echo Please make sure your virtual environment is installed.
    echo.
    pause
    exit /b 1
)

echo [OK] Using Python: !PYTHON_EXE!
echo [INFO] Running comparative evaluation on 5 multi-jurisdiction benchmark queries...
echo.

"!PYTHON_EXE!" -u compare_rag.py

echo.
echo ==============================================================================
echo  Benchmark complete! Full results saved to: benchmark_results.md
echo ==============================================================================
if exist benchmark_results.md (
    echo You can easily copy and review the full formatted answers from:
    echo   %~dp0benchmark_results.md
    echo.
)
pause >nul
