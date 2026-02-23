# ============================================
# run_local.ps1 - Headless Local Simulation
# Multi-seed + Results Directory (ICWS-ready)
# ============================================

Write-Host ">>> Activating virtual environment..."
$venv_paths = @(".\\.venv\\Scripts\\Activate.ps1", ".\\venv\\Scripts\\Activate.ps1")
$venv_found = $false
foreach ($path in $venv_paths) {
    if (Test-Path $path) {
        & $path
        $venv_found = $true
        break
    }
}
if (-not $venv_found) {
    Write-Host "[WARNING] Virtual environment not found. Assuming Python is in PATH."
}

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ROOT
Write-Host ">>> Project root = $ROOT"

# ------------------------------------------------
# Global env
# ------------------------------------------------
$env:COMM_SIM_DIR          = "comm_sim"
$env:INSTANCES_FILE        = "instances.json"
$env:FUNC_MAP_FILE         = "func_map.json"
$env:CUDA_VISIBLE_DEVICES  = ""

$env:USE_HTTP_EXEC = "0"
$env:LOCAL_COMPUTE = "1"

$env:DEFAULT_NET_LATENCY_MS = "50.0"
$env:USE_TRACE_CALIB        = "1"

# Local transient failures (serverless realism)
$env:LOCAL_OOM_PROB  = "0.01"   # main experiments: small but non-zero
$env:LOCAL_BUSY_PROB = "0.00"   # usually keep 0 unless stress testing

# 重试退避（更像真实 serverless）
$env:RETRY_BACKOFF_MS      = "5"
$env:RETRY_BACKOFF_MAX_MS  = "50"

# Network latency sanity protection
$env:MIN_NET_LATENCY_RATIO = "0.05"

$env:KEEP_ALIVE_MS        = "5000"
$env:EVICTION_BASE_PROB   = "0.02"
$env:EVICTION_TAU_MS      = "20000"

#拉大压力
$env:EXP_MAX_CONCURRENCY="1"
$env:HOT_PROB="0.75"
$env:COLD_EXPERT_LOAD_MS="400"

if (-not $env:MAX_STEPS) {
    $env:MAX_STEPS = "2000"
}

# ------------------------------------------------
# Result root directory (NEW)
# ------------------------------------------------
$RESULT_ROOT = Join-Path $ROOT "results"
if (-not (Test-Path $RESULT_ROOT)) {
    New-Item -ItemType Directory -Path $RESULT_ROOT | Out-Null
}

# 每次 run 一个独立目录（时间戳）
$TS = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$RUN_DIR = Join-Path $RESULT_ROOT $TS
New-Item -ItemType Directory -Path $RUN_DIR | Out-Null

Write-Host ">>> Results will be saved to:"
Write-Host "    $RUN_DIR" -ForegroundColor Yellow

# ------------------------------------------------
# Menu
# ------------------------------------------------
Write-Host "================================================"
Write-Host " Serverless MoE Local Simulation"
Write-Host " Multi-Seed + Result Directory"
Write-Host "================================================"
Write-Host " [1] Run Ours (Full)"
Write-Host " [2] Run Ablations"
Write-Host " [3] Run Baselines"
Write-Host " [4] Run ALL (Paper Set)"
Write-Host "================================================"
Write-Host " [S] Single seed"
Write-Host " [M] Multi-seed (0,1,2)"
Write-Host "================================================"

$choice = Read-Host "Enter choice (e.g. 4 M)"

$parts = $choice.Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
$exp_choice = $parts[0]
$seed_mode = if ($parts.Length -ge 2) { $parts[1].ToUpper() } else { "M" }

# Seeds
if ($seed_mode -eq "S") {
    $seed_in = Read-Host "Enter SEED (default 0)"
    if ([string]::IsNullOrWhiteSpace($seed_in)) { $seed_in = "0" }
    $seeds = @([int]$seed_in)
} else {
    $seeds = @(0,1,2)
}

# ------------------------------------------------
# Cleanup
# ------------------------------------------------
function Cleanup-Data {
    if (Test-Path "comm_sim") {
        Remove-Item -Path "comm_sim" -Recurse -Force -ErrorAction SilentlyContinue
    }
    Get-ChildItem -Path . -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 200
}

# ------------------------------------------------
# Run controller
# ------------------------------------------------
function Run-Controller {
    param([string]$etype, [string]$mode, [int]$seed)

    $env:SEED = $seed.ToString()
    Cleanup-Data

    # 子目录（按实验类型）
    $subdir = Join-Path $RUN_DIR $etype
    if (-not (Test-Path $subdir)) {
        New-Item -ItemType Directory -Path $subdir | Out-Null
    }

    if ($etype -eq "baseline") {
        $env:EXPERIMENT_TYPE = "baseline"
        $env:BASELINE_MODE  = $mode
        $env:METRICS_FILE   = Join-Path $subdir ("metrics_baseline_{0}_seed{1}.csv" -f $mode, $seed)
    } else {
        $env:EXPERIMENT_TYPE = "ablation"
        $env:ABLATION_MODE  = $mode
        $env:METRICS_FILE   = Join-Path $subdir ("metrics_{0}_seed{1}.csv" -f $mode, $seed)
    }

    Write-Host ">>> Running $etype / $mode / seed=$seed"
    Write-Host ">>> Output: $env:METRICS_FILE" -ForegroundColor Green

    python controller.py

    Write-Host ">>> Finished $etype / $mode / seed=$seed" -ForegroundColor Cyan
    Write-Host "------------------------------------------------"
}

# ------------------------------------------------
# Dispatch
# ------------------------------------------------
function Run-Choice {
    param([string]$c, [int]$seed)

    switch ($c) {
        "1" { Run-Controller "ablation" "full" $seed }
        "2" {
            foreach ($m in @("no_nsga","no_online","no_hotcold")) {
                Run-Controller "ablation" $m $seed
            }
        }
        "3" {
            foreach ($m in @("round_robin","random")) {
                Run-Controller "baseline" $m $seed
            }
        }
        "4" {
            Run-Controller "ablation" "full" $seed
            foreach ($m in @("round_robin","random")) {
                Run-Controller "baseline" $m $seed
            }
            foreach ($m in @("no_hotcold","no_nsga","no_online")) {
                Run-Controller "ablation" $m $seed
            }
        }
        Default { Write-Host "Invalid selection." }
    }
}

foreach ($s in $seeds) {
    Write-Host "=============================="
    Write-Host " Running seed = $s"
    Write-Host "=============================="
    Run-Choice $exp_choice $s
}

Write-Host "All runs finished."
Write-Host "Results saved under:"
Write-Host "  $RUN_DIR" -ForegroundColor Yellow
