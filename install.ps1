param([string]$Source = $env:FAKE_THERMAL_SOURCE, [string]$Ref = $(if ($env:FAKE_THERMAL_REF) { $env:FAKE_THERMAL_REF } else { 'main' }), [switch]$Turbo, [switch]$SkipPlugins, [switch]$Uninstall)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = if ($env:FAKE_THERMAL_HOME) { $env:FAKE_THERMAL_HOME } else { Join-Path $env:LOCALAPPDATA 'fake-thermal' }
$bootstrap = Join-Path $root 'bootstrap'
New-Item -ItemType Directory -Force $bootstrap | Out-Null
$target = 'x86_64-pc-windows-msvc'
if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $target = 'aarch64-pc-windows-msvc' }
Invoke-WebRequest -UseBasicParsing "https://github.com/astral-sh/uv/releases/download/0.9.18/uv-$target.zip" -OutFile "$bootstrap\uv.zip"
Expand-Archive -Force "$bootstrap\uv.zip" $bootstrap
$uv = (Get-ChildItem $bootstrap -Filter uv.exe -Recurse | Select-Object -First 1).FullName
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
& $uv python install 3.11
if ($LASTEXITCODE) { throw 'Python bootstrap failed' }
& $uv venv --python 3.11 --managed-python --allow-existing "$root\venv"
if ($LASTEXITCODE) { throw 'venv bootstrap failed' }
if (!$Source) {
    Invoke-WebRequest -UseBasicParsing "https://codeload.github.com/Alx8g/fake-thermal/zip/$Ref" -OutFile "$bootstrap\source.zip"
    $extract = Join-Path $bootstrap 'checkout'
    if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
    Expand-Archive "$bootstrap\source.zip" $extract
    $Source = (Get-ChildItem $extract -Directory | Select-Object -First 1).FullName
}
$arguments = @("$Source\scripts\install_user.py", '--root', $root, '--uv', $uv, '--source', $Source)
if ($Turbo) { $arguments += '--turbo' }
if ($SkipPlugins) { $arguments += '--skip-plugins' }
if ($Uninstall) { $arguments += '--uninstall' }
& "$root\venv\Scripts\python.exe" @arguments
if ($LASTEXITCODE) { throw 'Installation failed' }
if ($Uninstall) { Remove-Item -LiteralPath $root -Recurse -Force }
