@echo off
rem Switch the console to UTF-8 (see the ASCII note in start.cmd).
chcp 65001 >nul
echo [qoder2api] Checking for processes on port 8963...

set FOUND=0
rem Find the PID listening on 8963 via netstat. Never kill python.exe
rem globally - only the exact PID, so nothing else is harmed.
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8963" ^| findstr "LISTENING"') do (
    set FOUND=1
    echo [qoder2api] Found service process PID: %%a, terminating...
    taskkill /f /pid %%a >nul 2>&1
)

if "%FOUND%"=="1" (
    echo [qoder2api] Service stopped.
) else (
    echo [qoder2api] No process is listening on port 8963.
)

timeout /t 2 >nul
