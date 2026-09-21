@echo off
rem Switch the console to UTF-8 so non-ASCII output does not get garbled.
rem KEEP THIS FILE PURE ASCII: a .cmd that mixes chcp 65001 with non-ASCII
rem text can make cmd.exe lose its read position and hang silently (no
rem output, process never exits).
chcp 65001 >nul
cd /d "%~dp0"

rem Environment: httpx on Windows picks up the system proxy and times out,
rem so NO_PROXY must be forced to *.
set NO_PROXY=*
set QODER_HOST=127.0.0.1
set QODER_PORT=8963

echo [qoder2api] Starting the Qoder bridge service...
echo [qoder2api] Listening on: http://%QODER_HOST%:%QODER_PORT%
echo [qoder2api] Press Ctrl+C to stop the service

rem Launch the service with the project venv's own Python.
"%~dp0.venv\Scripts\python.exe" openai_bridge.py

pause
