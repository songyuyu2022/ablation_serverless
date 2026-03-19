Write-Host ">>> Activating virtual environment..."

# -----------------------------
# Activate venv (Windows)
# -----------------------------
$venvFound = $false

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
    $venvFound = $true
}
elseif (Test-Path ".\venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
    $venvFound = $true
}

if (-not $venvFound) {
    Write-Host "[WARNING] Virtual environment not found. Assuming Python is in PATH."
}

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ROOT
Write-Host ">>> Project root = $ROOT"

# ------------------------------------------------
# Global env
# ------------------------------------------------
$env:COMM_SIM_DIR="comm_sim"
$env:INSTANCES_FILE="instances.json"
$env:FUNC_MAP_FILE="func_map.json"

$env:CUDA_VISIBLE_DEVICES="0"

$env:USE_HTTP_EXEC="0"
$env:LOCAL_COMPUTE="1"

$env:DEFAULT_NET_LATENCY_MS="50.0"
$env:USE_TRACE_CALIB="1"

$env:LOCAL_OOM_PROB="0.01"
$env:LOCAL_BUSY_PROB="0.00"

$env:RETRY_BACKOFF_MS="5"
$env:RETRY_BACKOFF_MAX_MS="50"

$env:MIN_NET_LATENCY_RATIO="0.05"

$env:EVICTION_BASE_PROB="0.09"
$env:NUM_EXPERTS="16"
$env:FAST_LOCAL_BYPASS_COMM="1"
$env:SEQ_LEN="512"
$env:MICRO_BATCH="4"
$env:BATCH_SIZE="16"
$env:MAX_INFLIGHT_MICROBATCH="4"
$env:MAX_INFLIGHT_EXPERT="8"
$env:EXP_MAX_CONCURRENCY="4"
$env:HOT_PROB="0.45"
$env:COLD_EXPERT_LOAD_MS="400"

$env:DEVICE="cuda"
$env:AMP_ENABLED="1"
$env:AMP_DTYPE="fp16"
$env:SERIALIZE_TENSORS="0"

# Hot/Cold
$env:HOTSET_SIZE="2"
$env:HOT_ENTER_P="0.28"
$env:HOT_EXIT_P="0.20"
$env:HOT_MIN_STAY_STEPS="15"
$env:HEATMAP_UPDATE_EVERY="2"
$env:HEATMAP_DECAY="0.97"
$env:HEATMAP_TREND_DECAY="0.85"

# Acc logging
$env:ACC_EVERY="1"
$env:ACC_FILL_NAN="0"

# Predictor
$env:PRED_TARGET="nocold"
$env:PRED_MONITOR_WINDOW="200"
$env:MAX_GPU_COMPUTE_MS="800.0"    # 释放被压制的计算时间波动
$env:KEEP_ALIVE_MS="1000"          # 极速回收实例
$env:EVICTION_TAU_MS="3000"        # 增加回收概率
$env:COLD_ACC_STEPS="10"           # 拉长冷专家休眠期

if (-not $env:MAX_STEPS) {
    $env:MAX_STEPS="2000"
}

# ------------------------------------------------
# Result root directory
# ------------------------------------------------
$RESULT_ROOT = Join-Path $ROOT "results"
New-Item -ItemType Directory -Force -Path $RESULT_ROOT | Out-Null

$TS = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$RUN_DIR = Join-Path $RESULT_ROOT $TS
New-Item -ItemType Directory -Force -Path $RUN_DIR | Out-Null

Write-Host ">>> Results will be saved to:"
Write-Host "    $RUN_DIR"

# ------------------------------------------------
# Menu
# ------------------------------------------------
Write-Host "================================================"
Write-Host " Serverless MoE Local Simulation"
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
$parts = $choice.Split(" ")

$exp_choice = $parts[0]
$seed_mode = if ($parts.Length -gt 1) { $parts[1].ToUpper() } else { "M" }

if ($seed_mode -eq "S") {
    $seed_in = Read-Host "Enter SEED (default 0)"
    if (-not $seed_in) { $seed_in = 0 }
    $SEEDS = @($seed_in)
} else {
    $SEEDS = @(0,1,2)
}

# ------------------------------------------------
# Cleanup
# ------------------------------------------------
function Cleanup-Data {
    if (Test-Path "comm_sim") {
        Remove-Item -Recurse -Force "comm_sim"
    }

    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    Start-Sleep -Milliseconds 200
}

# ------------------------------------------------
# Run controller
# ------------------------------------------------
function Run-Controller {
    param (
        [string]$etype,
        [string]$mode,
        [string]$seed
    )

    $env:SEED = $seed
    Cleanup-Data

    $subdir = Join-Path $RUN_DIR $etype
    New-Item -ItemType Directory -Force -Path $subdir | Out-Null

    if ($etype -eq "baseline") {
        $env:EXPERIMENT_TYPE="baseline"
        $env:BASELINE_MODE=$mode
        $env:METRICS_FILE = Join-Path $subdir "metrics_baseline_${mode}_seed${seed}.csv"
    }
    else {
        $env:EXPERIMENT_TYPE="ablation"
        $env:ABLATION_MODE=$mode
        $env:METRICS_FILE = Join-Path $subdir "metrics_${mode}_seed${seed}.csv"
    }

    Write-Host ">>> Running $etype / $mode / seed=$seed"
    Write-Host ">>> Output: $env:METRICS_FILE"

    $logFile = Join-Path $subdir "log_${etype}_${mode}_seed${seed}.txt"

    python controller.py 2>&1 | Tee-Object -FilePath $logFile

    Write-Host ">>> Finished $etype / $mode / seed=$seed"
    Write-Host "------------------------------------------------"
}

# ------------------------------------------------
# Dispatch
# ------------------------------------------------
function Run-Choice {
    param (
        [string]$c,
        [string]$seed
    )

    switch ($c) {
        "1" {
            Run-Controller "ablation" "full" $seed
        }
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
        Default {
            Write-Host "Invalid selection."
        }
    }
}

foreach ($s in $SEEDS) {
    Write-Host "=============================="
    Write-Host " Running seed = $s"
    Write-Host "=============================="
    Run-Choice $exp_choice $s
}

Write-Host "All runs finished."
Write-Host "Results saved under:"
Write-Host "  $RUN_DIR"