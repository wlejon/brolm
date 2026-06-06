<#
.SYNOPSIS
    Download Mistral-Small-3.1-24B-Instruct-2503 into brolm's gitignored
    weights/ directory. PowerShell port of download_mistral.sh.

.DESCRIPTION
    Mistral 3.1 is a VLM = a 24B Mistral text decoder + a Pixtral vision
    encoder. brolm support is being built up incrementally (see the mistral3_*
    headers as they land); the tokenizer (brolm::mistral::Tokenizer, fed by
    tekken.json) is in first.

    Size warning: the full bf16 checkpoint is ~48 GB (there is no smaller 3.1
    release). Until the decoder/vision loaders land, the only thing brolm
    consumes is the tokenizer — so by default this script fetches ONLY the
    tokenizer + config files (~15 MB), which is all the gated tokenizer test
    needs. Set $env:FULL = '1' to pull the entire repo (weights included).

    This is a gated model: accept the terms on its Hugging Face page while
    signed in before downloading, or `hf download` will 403.

    Requires the Hugging Face CLI (`hf`) on PATH and an authenticated session
    (`hf auth whoami` should succeed). Downloads resume automatically if rerun.

.PARAMETER DestDir
    Optional. Where to place the weights. Defaults to
    <repo>/weights/<model> (weights/ is gitignored).

.EXAMPLE
    scripts\download_mistral.ps1
    # tokenizer-only (~15 MB)

.EXAMPLE
    $env:FULL = '1'; scripts\download_mistral.ps1
    # pull the full ~48 GB checkpoint

.EXAMPLE
    $env:REPO = 'mistralai/Mistral-Small-3.1-24B-Base-2503'; scripts\download_mistral.ps1
#>

[CmdletBinding()]
param(
    [string]$DestDir
)

$ErrorActionPreference = 'Stop'
# We check $LASTEXITCODE on `hf` ourselves (as the bash version does), so don't
# let PowerShell turn the CLI's stderr / nonzero exits into terminating errors.
$PSNativeCommandUseErrorActionPreference = $false
# Force the hf CLI (Python) into UTF-8 I/O. Without this it crashes on Windows
# consoles whose legacy codepage (cp1252) can't encode the '✓' it prints on
# success: "'charmap' codec can't encode character '✓'".
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Repo = if ($env:REPO) { $env:REPO } else { 'mistralai/Mistral-Small-3.1-24B-Instruct-2503' }
$Full = ($env:FULL -eq '1')

# Resolve the repo root from this script's location, independent of CWD.
$RepoRoot = Split-Path -Parent $PSScriptRoot

# HF repo ids use '/', so take the last path segment for the default folder name.
$ModelName = ($Repo -split '/')[-1]
$Dest = if ($DestDir) { $DestDir } else { Join-Path $RepoRoot "weights\$ModelName" }

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    Write-Error "'hf' (Hugging Face CLI) not found on PATH. Install with: pip install -U huggingface_hub"
    exit 1
}

hf auth whoami *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "not authenticated. Run 'hf auth login' first."
    exit 1
}

Write-Host "Repo:        $Repo"
Write-Host "Destination: $Dest"
if ($Full) {
    Write-Host "Mode:        FULL (~48 GB: safetensors shards + tokenizer + config)"
} else {
    Write-Host "Mode:        tokenizer-only (~15 MB; set `$env:FULL = '1' for the weights)"
}
Write-Host ""

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# Whole repo when FULL, otherwise just the tokenizer + config files. brolm's
# loaders consume the safetensors shards plus tekken.json directly — no offline
# conversion. The gated tokenizer test reads tekken.json from $BROLM_MISTRAL_DIR
# (= this Dest).
$dlArgs = @('download', $Repo, '--local-dir', $Dest)
if (-not $Full) {
    $dlArgs += @(
        '--include', 'tekken.json',
        '--include', 'tokenizer_config.json',
        '--include', 'special_tokens_map.json',
        '--include', 'config.json'
    )
}

& hf @dlArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "hf download failed (exit $LASTEXITCODE)."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Done. $Repo is in: $Dest"
Write-Host ""
Write-Host "Run the gated tokenizer test against it with:"
Write-Host "  `$env:BROLM_MISTRAL_DIR = '$Dest'; ctest --test-dir build -C Release -R mistral_tokenizer"
