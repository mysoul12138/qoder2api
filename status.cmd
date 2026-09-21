@echo off
rem Switch the console to UTF-8 (see the ASCII note in start.cmd).
chcp 65001 >nul
cd /d "%~dp0"

echo [qoder2api] Checking service status...
netstat -aon | findstr ":8963" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [qoder2api] Status: running ^(port 8963 is listening^)
    echo [qoder2api] Testing model list fetch...
    curl -s http://127.0.0.1:8963/v1/models
    echo.
) else (
    echo [qoder2api] Status: not running
)
pause
