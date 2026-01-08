$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$configPath = Join-Path $repoRoot "windows_native\\invoice_config.json"
if (Test-Path $configPath) {
  $cfg = Get-Content $configPath -Raw | ConvertFrom-Json
  if ($cfg.llm -and $cfg.llm.api_key -and $cfg.llm.api_key.Trim().Length -gt 0) {
    throw "invoice_config.json enthaelt einen API-Key. Bitte entfernen, bevor du bundlest."
  }
}

python -m PyInstaller `
  --noconfirm `
  --clean `
  "InvoiceApp.spec" #spec file is the configuration file for the PyInstaller

$distDir = Join-Path $repoRoot "dist" 
$zipPath = Join-Path $distDir "InvoiceApp.zip" #Join
if (-not (Test-Path $distDir)) {
  throw "dist-Verzeichnis fehlt nach dem Build."
}
if (Test-Path $zipPath) {
  Remove-Item $zipPath -Force
}
Compress-Archive -Path (Join-Path $distDir "*") -DestinationPath $zipPath
