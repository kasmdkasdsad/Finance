@echo off
rem Stop QuantPulse: asks the API and the UI the launcher started to shut down cleanly (Ctrl+C),
rem forcing them only if they have not stopped after 20 seconds.
call "%~dp0launch.bat" stop
