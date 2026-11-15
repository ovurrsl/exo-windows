# Allow inbound UDP 52413 (discovery) and TCP for exo / python.exe on Windows Firewall.
# PAIR's Windows installer does the equivalent for its proxy ports.

#Requires -RunAsAdministrator

param(
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$TcpRuleName = 'exo local cluster'
$UdpRuleName = 'exo local cluster UDP'
$PythonRuleName = 'exo python TCP'

foreach ($name in @($TcpRuleName, $UdpRuleName, $PythonRuleName)) {
    if (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $name
    }
}

New-NetFirewallRule -DisplayName $TcpRuleName -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort 52413,52414,52415 | Out-Null
New-NetFirewallRule -DisplayName $UdpRuleName -Direction Inbound -Action Allow `
    -Protocol UDP -LocalPort 52413 | Out-Null

$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$Candidates = @()
if ($PythonPath) {
    $Candidates += $PythonPath
}
$Candidates += @(
    (Join-Path $Root '.venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE 'AppData\Roaming\uv\python\cpython-3.13.14-windows-x86_64-none\python.exe')
)
$PythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue
if ($PythonCmd) {
    $Candidates += $PythonCmd.Source
}
$ResolvedPython = $Candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if ($ResolvedPython) {
    New-NetFirewallRule -DisplayName $PythonRuleName -Direction Inbound -Action Allow `
        -Protocol TCP -Program $ResolvedPython | Out-Null
    Write-Host "Firewall rules allow TCP 52413-52415, UDP 52413, and inbound TCP for '$ResolvedPython'."
} else {
    Write-Warning "Could not locate python.exe; added port rules for TCP 52413-52415 and UDP 52413 only. Re-run with -PythonPath <path-to-python.exe> for dynamic ring TCP ports."
    Write-Host "Firewall rules allow TCP 52413-52415 and UDP 52413 inbound."
}
