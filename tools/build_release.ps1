# LUMEN RAW release build: tests -> DirectML check -> PyInstaller -> portable ZIP ->
# Inno Setup installer -> installer extraction check -> smoke tests -> SHA256SUMS.
# Run from Explorer with build-release.cmd, or:
#   powershell -ExecutionPolicy Bypass -File tools\build_release.ps1 [-PythonPath <python.exe>] [-Iscc <ISCC.exe>] [-SampleRaw <photo>]
param([string]$PythonPath = '', [string]$Iscc = '', [string]$SampleRaw = '', [switch]$SkipTests)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$version = ([regex]::Match((Get-Content -Raw -LiteralPath 'lumen\__init__.py'), "__version__ = '([0-9.]+)'")).Groups[1].Value
$tag = 'v' + $version.Replace('.', '')
$publish = Join-Path $root ".publish\$tag"
$logs = Join-Path $publish 'logs'
$packages = Join-Path $publish 'packages'
New-Item -ItemType Directory -Force -Path $logs, $packages | Out-Null
$status = Join-Path $logs 'status.txt'
$buildLog = Join-Path $logs 'build.log'
Set-Content -LiteralPath $buildLog -Value "LUMEN RAW $version release build $(Get-Date -Format s)" -Encoding UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

function Say([string]$text) {
    Write-Host $text
    Add-Content -LiteralPath $buildLog -Value $text -Encoding UTF8
}
function Step([string]$name) {
    Say ''
    Say "==== $name  [$(Get-Date -Format HH:mm:ss)] ===="
    Set-Content -LiteralPath $status -Value "running: $name" -Encoding UTF8
}
function Fail([string]$what) {
    Say "FAILED: $what"
    Set-Content -LiteralPath $status -Value "FAILED: $what" -Encoding UTF8
    throw "FAILED: $what"
}
function Run([string]$what, [string]$exe, [string[]]$arguments, [string]$log = '') {
    if (-not $log) { $log = $buildLog }
    # Tee-Object writes UTF-16 on Windows PowerShell 5.1; append UTF-8 lines instead.
    & $exe @arguments 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -LiteralPath $log -Value $line -Encoding UTF8 }
    if ($LASTEXITCODE -ne 0) { Fail "$what (exit $LASTEXITCODE)" }
}

try {
    Step 'Python environment'
    $baseProbe = "import PySide6, rawpy, cv2, tifffile, numpy, PyInstaller, pytest, onnxruntime; print('base ok')"
    $runtimeProbe = "import onnxruntime as o, windowsml; v = tuple(int(x) for x in o.__version__.split('.')[:2]); assert v >= (1, 30), o.__version__; assert 'DmlExecutionProvider' in o.get_available_providers(), o.get_available_providers(); print('onnxruntime', o.__version__, o.get_available_providers())"
    $python = $null
    foreach ($candidate in @($PythonPath, $env:LUMEN_PYTHON, '.venv\Scripts\python.exe')) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            & $candidate -c $baseProbe 2>&1 | ForEach-Object { "$_" } | Out-Host
            if ($LASTEXITCODE -eq 0) { $python = (Resolve-Path -LiteralPath $candidate).Path; break }
            Say "  skipped $candidate (missing GUI / build packages)"
        }
    }
    if (-not $python) {
        Say 'Creating .venv with Python 3.12 (requires internet access to PyPI)...'
        Run 'create .venv' 'py' @('-3.12', '-m', 'venv', '.venv')
        $python = (Resolve-Path -LiteralPath '.venv\Scripts\python.exe').Path
        Run 'install locked dependencies' $python @('-m', 'pip', 'install', '--no-cache-dir', '-r', 'requirements-lock.txt')
    }
    Say "Python: $python"
    & $python -c $runtimeProbe 2>&1 | ForEach-Object { "$_" } | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Say 'Switching the environment to the Windows ML runtime (onnxruntime-windowsml + windowsml)...'
        Run 'record previous packages' $python @('-m', 'pip', 'freeze') (Join-Path $logs 'pip-freeze-before.txt')
        & $python -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu onnxruntime-windowsml 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -LiteralPath $buildLog -Value $line -Encoding UTF8 }
        Run 'install Windows ML runtime' $python @('-m', 'pip', 'install', '--no-cache-dir', '-r', 'requirements-directml.txt')
        Run 'probe Windows ML runtime' $python @('-c', $runtimeProbe)
    }
    Run 'freeze environment' $python @('-m', 'pip', 'freeze') (Join-Path $logs 'pip-freeze.txt')

    Step 'Runtime assets'
    Run 'verify runtime assets' $python @('tools\fetch_assets.py', '--verify-only')

    Step 'DirectML GPU graphs'
    Run 'GPU node execution check' $python @('tools\check_gpu.py')

    if (-not $SkipTests) {
        Step 'Regression tests'
        $env:QT_QPA_PLATFORM = 'offscreen'
        $basetemp = Join-Path $publish 'pytest-temp'
        if (Test-Path -LiteralPath $basetemp) { Remove-Item -LiteralPath $basetemp -Recurse -Force }
        Run 'pytest' $python @('-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider', "--basetemp=$basetemp") (Join-Path $logs 'pytest.log')
        Remove-Item Env:\QT_QPA_PLATFORM
    }

    Step 'Sample photo'
    $sample = $null
    foreach ($candidate in @($SampleRaw, 'F:\DCIM\100MSDCF\DSC07998.ARW', 'F:\DCIM\100MSDCF\DSC08026.ARW')) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { $sample = (Resolve-Path -LiteralPath $candidate).Path; break }
    }
    if (-not $sample) {
        $sample = Join-Path $publish 'synthetic-sample.png'
        Run 'synthetic sample' $python @('-c', "import sys; sys.path.insert(0, '.'); import numpy as np, cv2; from tools.bench_pipeline import sample; cv2.imwrite(r'$sample', (np.clip(sample(None), 0, 1) ** (1 / 2.2) * 65535).astype(np.uint16)[..., ::-1])")
        Say 'No camera RAW found; using a synthetic 16-bit PNG.'
    }
    Say "Sample: $sample"
    Run 'pipeline benchmark' $python @('tools\bench_pipeline.py', $sample, (Join-Path $logs 'benchmark.json'))
    $smoke = Join-Path $publish 'source-smoke'
    if (Test-Path -LiteralPath $smoke) { Remove-Item -LiteralPath $smoke -Recurse -Force }
    Run 'source smoke test' $python @('main.py', '--smoke-test', $sample, $smoke)

    Step 'PyInstaller (DirectML onedir)'
    Run 'PyInstaller' $python @('-m', 'PyInstaller', '--noconfirm', '--clean', 'LumenRAW.spec') (Join-Path $logs 'pyinstaller.log')
    if (-not (Test-Path -LiteralPath 'dist\LumenRAW\LumenRAW.exe')) { Fail 'frozen executable missing' }

    Step 'Frozen smoke test'
    $frozenSmoke = Join-Path $publish 'portable-smoke'
    if (Test-Path -LiteralPath $frozenSmoke) { Remove-Item -LiteralPath $frozenSmoke -Recurse -Force }
    $process = Start-Process -FilePath 'dist\LumenRAW\LumenRAW.exe' -ArgumentList @('--smoke-test', "`"$sample`"", "`"$frozenSmoke`"") -Wait -PassThru
    if ($process.ExitCode -ne 0) { Fail "frozen smoke test (exit $($process.ExitCode)); see $frozenSmoke\report.json" }
    Get-Content -LiteralPath (Join-Path $frozenSmoke 'report.json') -Encoding UTF8 | Where-Object { $_ -match '"(ok|backend|provider|device|warning)"' } | ForEach-Object { Say $_ }
    # 1.6.0: the console lumen-cli.exe (command line and MCP server).
    if (-not (Test-Path -LiteralPath 'dist\LumenRAW\lumen-cli.exe')) { Fail 'lumen-cli.exe missing' }
    Run 'frozen lumen-cli and MCP check' $python @('tools\check_cli.py', 'dist\LumenRAW\lumen-cli.exe', $sample, (Join-Path $publish 'cli-smoke')) (Join-Path $logs 'cli-smoke.log')

    Step 'Portable ZIP'
    foreach ($old in @("LumenRAW-$version-Windows.zip", "LumenRAW-$version-Windows.json", "LumenRAW-$version-Windows.zip.tmp")) {
        $path = Join-Path $packages $old
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
    Run 'package portable ZIP' $python @('tools\package_windows.py')

    Step 'Inno Setup installer'
    if (-not $Iscc) {
        $command = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
        $options = @()
        if ($command) { $options += $command.Source }
        $options += @("${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe", "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
                      "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                      "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")
        $Iscc = $options | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    }
    if (-not $Iscc) { Fail 'ISCC.exe not found; install Inno Setup 7 or pass -Iscc' }
    Say "ISCC: $Iscc"
    Run 'compile installer' $Iscc @('/Qp', '/DAppBuild=dist\LumenRAW', "/DAppVersion=$version", 'installer.iss') (Join-Path $logs 'iscc.log')
    $setup = Join-Path $packages "LumenRAW-$version-Setup.exe"
    if (-not (Test-Path -LiteralPath $setup)) { Fail 'installer missing' }

    Step 'Installer extraction check'
    $extract = Join-Path $publish 'installer-extract'
    if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force }
    $process = Start-Process -FilePath $setup -ArgumentList @('/PORTABLE=1', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/NOICONS', "/DIR=`"$extract`"", "/LOG=`"$(Join-Path $logs 'installer-extract.log')`"") -Wait -PassThru
    if ($process.ExitCode -ne 0) { Fail "installer extraction (exit $($process.ExitCode))" }
    Run 'compare installer with frozen build' $python @('tools\verify_installer.py', $extract) (Join-Path $packages "LumenRAW-$version-Installer-verified.json")
    $installedSmoke = Join-Path $publish 'installer-smoke'
    if (Test-Path -LiteralPath $installedSmoke) { Remove-Item -LiteralPath $installedSmoke -Recurse -Force }
    $process = Start-Process -FilePath (Join-Path $extract 'LumenRAW.exe') -ArgumentList @('--smoke-test', "`"$sample`"", "`"$installedSmoke`"") -Wait -PassThru
    if ($process.ExitCode -ne 0) { Fail "installer smoke test (exit $($process.ExitCode))" }
    Run 'installed lumen-cli and MCP check' $python @('tools\check_cli.py', (Join-Path $extract 'lumen-cli.exe'), $sample, (Join-Path $publish 'installer-cli-smoke')) (Join-Path $logs 'installer-cli-smoke.log')

    Step 'Checksums'
    Run 'SHA256SUMS' $python @('tools\package_windows.py', '--checksums')
    Get-ChildItem -LiteralPath $packages -File | ForEach-Object { Say ("{0,14:N0}  {1}" -f $_.Length, $_.Name) }
    Set-Content -LiteralPath $status -Value "SUCCESS $version $(Get-Date -Format s)" -Encoding UTF8
    Say ''
    Say "SUCCESS: LUMEN RAW $version packages are in $packages"
    exit 0
}
catch {
    Say "$_"
    if (-not ((Get-Content -Raw -LiteralPath $status) -like 'FAILED*')) {
        Set-Content -LiteralPath $status -Value "FAILED: $_" -Encoding UTF8
    }
    exit 1
}
