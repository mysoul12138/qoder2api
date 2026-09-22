@echo off
rem Add a Qoder PAT to the account pool (pool.json) with live gateway verify.
rem KEEP THIS FILE PURE ASCII: chcp 65001 + non-ASCII bytes can hang cmd.exe.
chcp 65001 >nul
cd /d "%~dp0"

rem httpx on Windows must not pick up the system proxy for the verify call.
set NO_PROXY=*

"%~dp0.venv\Scripts\python.exe" add_pat.py %*
if errorlevel 1 echo [add-pat] exited with error.

echo.
pause
