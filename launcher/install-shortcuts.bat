@echo off
rem Creates the QuantPulse desktop shortcuts (run once; run again any time to repair them).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-shortcuts.ps1" %*
if errorlevel 1 pause
