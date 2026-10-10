<#
.SYNOPSIS
    Creates the QuantPulse desktop shortcuts. Run once (double-click install-shortcuts.bat); run again any
    time to repair them.

.DESCRIPTION
    * "QuantPulse Terminal"        starts the API and the UI and opens the browser.
    * "QuantPulse Trading Control" does the same and opens the Alpaca PAPER trading page.
    * "Stop QuantPulse"            stops them cleanly.

    Every shortcut runs launcher\quantpulse_launcher.py with the project's own virtual environment
    (.venv\Scripts\pythonw.exe: no console window). None of them runs a strategy cycle, places an order or
    changes a setting. Paths are absolute, so the shortcuts work from anywhere.
#>
[CmdletBinding()]
param(
    [switch]$NoStopShortcut,   # skip "Stop QuantPulse"
    [switch]$Quiet             # no message box at the end
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$Title = 'QuantPulse Terminal'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Stop-WithError([string]$Message) {
    Write-Host $Message -ForegroundColor Red
    [void][System.Windows.Forms.MessageBox]::Show($Message, $Title, 'OK', 'Error')
    exit 1
}

foreach ($rel in @('pyproject.toml', 'src\quantpulse\__init__.py', 'frontend\app.py',
                   'launcher\quantpulse_launcher.py', 'assets\quantpulse.ico')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $rel))) {
        Stop-WithError "QuantPulse is incomplete: '$rel' was not found in $Root."
    }
}
$Pythonw = Join-Path $Root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $Pythonw)) {
    Stop-WithError ("The QuantPulse virtual environment was not found:`n$Root\.venv`n`n" +
        "Create it once in PowerShell:`n  cd $Root`n  py -3.11 -m venv .venv`n" +
        "  .venv\Scripts\pip install -e "".[frontend]"" -c constraints.txt`n`nThen run this again.")
}

# [Environment]::GetFolderPath is right even when the Desktop is redirected (e.g. to OneDrive).
$Desktop = [Environment]::GetFolderPath('Desktop')
$Launcher = Join-Path $Root 'launcher\quantpulse_launcher.py'
$Shell = New-Object -ComObject WScript.Shell
$Created = @()

function New-QuantPulseShortcut([string]$Name, [string]$Arguments, [string]$Icon, [string]$Description) {
    $path = Join-Path $Desktop "$Name.lnk"
    $lnk = $Shell.CreateShortcut($path)
    $lnk.TargetPath = $Pythonw
    $lnk.Arguments = """$Launcher"" $Arguments"
    $lnk.WorkingDirectory = $Root
    $lnk.IconLocation = (Join-Path $Root "assets\$Icon") + ',0'
    $lnk.WindowStyle = 1
    $lnk.Description = $Description
    $lnk.Save()
    Write-Host "Created $path"
    $script:Created += $path
}

New-QuantPulseShortcut 'QuantPulse Terminal' 'start' 'quantpulse.ico' `
    'Start QuantPulse Terminal (Alpaca PAPER trading). Starting never places a trade.'
New-QuantPulseShortcut 'QuantPulse Trading Control' 'start --page trading' 'quantpulse-trading.ico' `
    'Open the Alpaca PAPER trading page: account, positions, orders, risk, kill switch. Never trades by itself.'
if (-not $NoStopShortcut) {
    New-QuantPulseShortcut 'Stop QuantPulse' 'stop' 'quantpulse-stop.ico' 'Stop the QuantPulse API and UI.'
}

if (-not $Quiet) {
    [void][System.Windows.Forms.MessageBox]::Show(
        ("Desktop shortcuts created:`n`n" + (($Created | ForEach-Object { '  ' + (Split-Path $_ -Leaf) }) -join "`n") +
         "`n`nin $Desktop"), $Title, 'OK', 'Information')
}
