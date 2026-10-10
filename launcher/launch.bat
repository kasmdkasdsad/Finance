@echo off
rem QuantPulse Terminal: starts the API and the UI (if not already running) and opens the browser.
rem Starting QuantPulse never runs a strategy cycle and never places an order.
rem   launch.bat                 start QuantPulse
rem   launch.bat start --page trading
rem   launch.bat stop            stop what the launcher started
rem   launch.bat status          show what is running
setlocal EnableExtensions DisableDelayedExpansion
for %%I in ("%~dp0..") do set "QP_ROOT=%%~fI"
set "QP_PY=%QP_ROOT%\.venv\Scripts\python.exe"
set "QP_PYW=%QP_ROOT%\.venv\Scripts\pythonw.exe"
set "QP_LAUNCHER=%QP_ROOT%\launcher\quantpulse_launcher.py"

if not exist "%QP_ROOT%\pyproject.toml" goto :no_project
if not exist "%QP_ROOT%\src\quantpulse" goto :no_project
if not exist "%QP_PY%" goto :no_venv
if not exist "%QP_PYW%" set "QP_PYW=%QP_PY%"

rem pythonw: no console window; the launcher shows its own start-up window and message boxes.
start "QuantPulse Terminal" "%QP_PYW%" "%QP_LAUNCHER%" %*
exit /b 0

:no_project
set "QP_MSG=QuantPulse was not found at %QP_ROOT%.|Keep the launcher folder inside the QuantPulse project folder."
goto :fail

:no_venv
set "QP_MSG=The QuantPulse virtual environment was not found:|%QP_ROOT%\.venv||Create it once in PowerShell:|  cd %QP_ROOT%|  py -3.11 -m venv .venv|  .venv\Scripts\pip install -e ".[frontend]" -c constraints.txt"
goto :fail

:fail
echo QuantPulse Terminal could not start:
echo %QP_MSG:|= %
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show(($env:QP_MSG -replace '\|', [Environment]::NewLine), 'QuantPulse Terminal', 'OK', 'Error')"
exit /b 1
