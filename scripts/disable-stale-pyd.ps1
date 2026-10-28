param(
    [string]$SitePackages = (Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Lib\site-packages'),
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'

$targets = @(
    'checkpoint\ftw'
    'engine\engine'
    'launch'
    'models\gemma4\gguf'
    'models\loader'
    'models\nvfp4_banks'
    'models\qwen3_5_moe\weight'
    'models\weight'
    'moe\cpu_executor'
    'moe\expert_banks'
    'moe\fused'
    'moe\host_banks'
    'moe\offload_cache'
    'scheduler\config'
    'scheduler\scheduler'
    'server\api_server'
    'server\args'
)

$packageRoot = Join-Path $SitePackages 'freetoken'
if (-not (Test-Path -LiteralPath $packageRoot -PathType Container)) {
    throw "FreeToken package directory does not exist: $packageRoot"
}

Write-Output "Allowlist contains exactly $($targets.Count) FreeToken modules."
foreach ($target in $targets) {
    $modulePath = Join-Path $packageRoot $target
    $moduleDirectory = Split-Path -Parent $modulePath
    $moduleName = Split-Path -Leaf $modulePath
    $sourcePath = "$modulePath.py"

    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        Write-Warning "Expected Python source is missing; refusing to disable anything for $target."
        continue
    }

    $extensions = @(
        Get-ChildItem -LiteralPath $moduleDirectory -File -Filter "$moduleName.*-win_amd64.pyd" |
            Where-Object { $_.Name -match "^$([regex]::Escape($moduleName))\.(?:cp\d+|abi3)-win_amd64\.pyd$" }
    )

    foreach ($extension in $extensions) {
        $disabledName = "$($extension.Name).disabled"
        $disabledPath = Join-Path $moduleDirectory $disabledName
        if (Test-Path -LiteralPath $disabledPath) {
            $suffix = Get-Date -Format 'yyyyMMddTHHmmssfff'
            $disabledName = "$($extension.Name).disabled.$suffix"
            $disabledPath = Join-Path $moduleDirectory $disabledName
            if (Test-Path -LiteralPath $disabledPath) {
                throw "Refusing to overwrite an existing disabled extension: $disabledPath"
            }
        }

        if ($Apply) {
            Rename-Item -LiteralPath $extension.FullName -NewName $disabledName
            Write-Output "Disabled: $target"
        }
        else {
            Write-Output "Would disable: $target as $disabledName"
        }
    }
}

if (-not $Apply) {
    Write-Output 'Preview only. Stop all FreeToken processes, review the allowlisted targets, then rerun with -Apply.'
}
