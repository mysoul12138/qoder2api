@echo off
rem Background helper for start-silent.vbs: redirect output to qoder2api.log.
rem Switch the console to UTF-8 (see the ASCII note in start.cmd).
chcp 65001 >nul
cd /d "%~dp0"
set NO_PROXY=*
set QODER_HOST=127.0.0.1
set QODER_PORT=8963
"%~dp0.venv\Scripts\python.exe" openai_bridge.py >> "%~dp0qoder2api.log" 2>&1
