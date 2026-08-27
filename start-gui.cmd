@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "gui.py"
  exit /b 0
)
where pyw >nul 2>nul
if %errorlevel%==0 (
  start "" pyw -3 "gui.py"
  exit /b 0
)
echo Python virtual environment was not found.
echo Run docs\SETUP.md first, then double-click start-gui.cmd again.
pause
exit /b 1
