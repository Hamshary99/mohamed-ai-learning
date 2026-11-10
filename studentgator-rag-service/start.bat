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
echo ============================================================
echo   StudentGator AI Advisory - Starting Backend Service
echo   Python:     %PYTHON_EXE%
echo   Swagger UI: http://localhost:8000/docs
echo   Web Chat:   http://localhost:8000/chat
echo ============================================================
echo Opening Swagger API docs in your default browser...

REM Open Swagger API UI in default browser immediately
start "" "http://localhost:8000/docs"

REM Launch FastAPI Server
"%PYTHON_EXE%" src\main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error. Press any key to close.
    pause >nul
)
