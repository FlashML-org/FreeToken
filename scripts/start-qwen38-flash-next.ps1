[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [int]$Port = 1927,
    [int]$MaxContext = 262144,
    [int]$MoeCacheSize = 1024
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe'
$kernelDir = Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Lib\site-packages\freetoken\kernel'

foreach ($requiredPath in @($ModelPath, $python, $kernelDir)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required FreeToken path does not exist: $requiredPath"
    }
}

$env:PYTHONPATH = Join-Path $repoRoot 'python'
$env:FREETOKEN_INSTALLED_KERNEL_DIR = $kernelDir
$env:FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK = '1'
$env:FREETOKEN_LOAD_VISION = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:FREETOKEN_PIN_BUDGET_GB = '64'

Write-Host "Starting Qwen3.8-Flash-Next NVFP4 on http://127.0.0.1:$Port"
Write-Host "OpenAI model: qwen3.8-flash-next-nvfp4; context: $MaxContext; vision: enabled"

& $python -m freetoken.cli serve `
    --model $ModelPath `
    --host 127.0.0.1 `
    --port $Port `
    --served-model-name qwen3.8-flash-next-nvfp4 `
    --dtype bfloat16 `
    --max-running-requests 1 `
    --max-seq-len-override $MaxContext `
    --num-tokens $MaxContext `
    --max-prefill-length 8192 `
    --attention-backend auto `
    --cache-type naive `
    --moe-backend offload `
    --nvfp4-backend triton `
    --expert-load serial `
    --moe-cache-size $MoeCacheSize `
    --moe-cpu-layers 0 `
    --tool-call-parser qwen `
    --reasoning-parser qwen3

exit $LASTEXITCODE
