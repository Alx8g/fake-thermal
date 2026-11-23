& {
$ErrorActionPreference = 'Stop'
$Source = $env:FAKE_THERMAL_SOURCE
$Ref = if ($env:FAKE_THERMAL_REF) { $env:FAKE_THERMAL_REF } else { 'main' }
$Turbo = $env:FAKE_THERMAL_TURBO -eq '1'
$SkipPlugins = $env:FAKE_THERMAL_SKIP_PLUGINS -eq '1'
$Uninstall = $env:FAKE_THERMAL_UNINSTALL -eq '1'
for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        '-Source' { $i++; if ($i -ge $args.Count) { throw '-Source requires a directory' }; $Source = $args[$i] }
        '-Ref' { $i++; if ($i -ge $args.Count) { throw '-Ref requires a ref' }; $Ref = $args[$i] }
        '-Turbo' { $Turbo = $true }
        '-SkipPlugins' { $SkipPlugins = $true }
        '-Uninstall' { $Uninstall = $true }
        default { throw "Unknown option: $($args[$i])" }
    }
}
$root = if ($env:FAKE_THERMAL_HOME) { $env:FAKE_THERMAL_HOME } else { Join-Path $env:LOCALAPPDATA 'fake-thermal' }
if ($Uninstall) {
    if (!(Test-Path $root)) { Write-Host 'fake-thermal is not installed.'; return }
    & "$root\venv\Scripts\python.exe" "$root\installer\scripts\install_user.py" --root $root --uv "$root\bootstrap\uv.exe" --source "$root\installer" --uninstall
    if ($LASTEXITCODE) { throw 'Uninstall failed' }
    Remove-Item -LiteralPath $root -Recurse -Force
    return
}
$oldProtocol = [Net.ServicePointManager]::SecurityProtocol
$oldPythonDir = $env:UV_PYTHON_INSTALL_DIR
try {
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$bootstrap = Join-Path $root 'bootstrap'
New-Item -ItemType Directory -Force $bootstrap | Out-Null
$target = 'x86_64-pc-windows-msvc'
if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $target = 'aarch64-pc-windows-msvc' }
Invoke-WebRequest -UseBasicParsing "https://github.com/astral-sh/uv/releases/download/0.9.18/uv-$target.zip" -OutFile "$bootstrap\uv.zip"
Expand-Archive -Force "$bootstrap\uv.zip" $bootstrap
$uv = (Get-ChildItem $bootstrap -Filter uv.exe -Recurse | Select-Object -First 1).FullName
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
& $uv python install --no-bin --no-registry 3.11
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
} finally {
    [Net.ServicePointManager]::SecurityProtocol = $oldProtocol
    $env:UV_PYTHON_INSTALL_DIR = $oldPythonDir
}
} @args
