@echo off
rem QuantPulse Trading Control: makes sure QuantPulse is running and opens the Alpaca PAPER trading page
rem (account, positions, orders, risk, proposed trades, kill switch, reconciliation). It never trades.
call "%~dp0launch.bat" start --page trading
