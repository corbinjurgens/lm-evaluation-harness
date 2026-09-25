param(
    [Parameter(Position = 0)]
    [string] $Model,
    [string] $BaseUrl,
    [string] $Image,
    [double] $Temperature,
    [int] $MaxGenToks,
    [ValidateSet("low", "medium", "high")]
    [string] $ReasoningEffort,
    [switch] $SmokeOnly,
    # Any other `scripts/benchmark.py run` flags, forwarded unchanged.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArguments
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$Harness = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Harness ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Host "Creating $Harness\.venv..."
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3 -m venv (Join-Path $Harness ".venv")
    }
    else {
        $SystemPython = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $SystemPython) {
            throw "Python 3.9 or newer is required. Install Python, then retry."
        }
        & $SystemPython.Source -m venv (Join-Path $Harness ".venv")
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python virtual-environment creation failed with exit code $LASTEXITCODE."
    }
}

$RunnerArguments = @()
if ($Model) {
    $RunnerArguments += $Model
}
if ($BaseUrl) {
    $RunnerArguments += @("--base-url", $BaseUrl)
}
if ($Image) {
    $RunnerArguments += @("--image", $Image)
}
if ($PSBoundParameters.ContainsKey("Temperature")) {
    $RunnerArguments += @("--temperature", $Temperature.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($PSBoundParameters.ContainsKey("MaxGenToks")) {
    $RunnerArguments += @("--max-gen-toks", $MaxGenToks.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($ReasoningEffort) {
    $RunnerArguments += @("--reasoning-effort", $ReasoningEffort)
}
if ($SmokeOnly) {
    $RunnerArguments += "--smoke-only"
}
if ($RemainingArguments) {
    $RunnerArguments += $RemainingArguments
}

& $Python (Join-Path $Harness "scripts\benchmark.py") run @RunnerArguments
exit $LASTEXITCODE
