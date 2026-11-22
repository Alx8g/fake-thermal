param([switch]$Uninstall)
$ErrorActionPreference = 'Stop'
$destination = Join-Path $env:APPDATA 'Adobe\CEP\extensions\com.alx8g.fakethermal.panel'
if ($Uninstall) {
    if (Test-Path $destination) { Remove-Item -LiteralPath $destination -Recurse -Force }
    Write-Host 'Fake Thermal uninstalled. Shared PlayerDebugMode settings were retained.'
    exit
}
New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($name in @('CSXS', 'css', 'js', 'jsx', 'index.html')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $destination -Recurse -Force
}
foreach ($version in 9..12) {
    $key = "HKCU:\Software\Adobe\CSXS.$version"
    New-Item -Path $key -Force | Out-Null
    New-ItemProperty -Path $key -Name PlayerDebugMode -Value '1' -PropertyType String -Force | Out-Null
}
Write-Host 'Installed. Restart Premiere, then open Window > Extensions (Legacy) > Fake Thermal.'
