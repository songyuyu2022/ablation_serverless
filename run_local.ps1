# ============================================
# run_ablation_all.ps1 - Automated Experiment Runner
# ============================================

Write-Host ">>> Activating virtual environment..."
$venv = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    & $venv
} else {
    Write-Host "[ERROR] Virtual environment not found! Expected at $venv"
    exit
}

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host ">>> Project root = $ROOT"

# --------------------------------------------
# Global env
# --------------------------------------------
$env:COMM_SIM_DIR        = "comm_sim"
$env:INSTANCES_FILE      = "instances.json"
$env:FUNC_MAP_FILE       = "func_map.json"
$env:CUDA_VISIBLE_DEVICES = ""      # CPU only

# --------------------------------------------
# Helper: start new PowerShell window
# --------------------------------------------
function Start-Window {
    param([string]$command)
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $command
}

# --------------------------------------------
# Start all function windows
# --------------------------------------------
function Start-AllFunctions {

    Write-Host "`n>>> Starting function windows..."

    # pre_fn
    Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8001"
    Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8002"
    Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8003"

    # post_fn
    Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8101"
    Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8102"
    Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8103"

    # experts
    Start-Window -command "`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8201"
    Start-Window -command "`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8202"
    Start-Window -command "`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8203"

    Start-Window -command "`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8211"
    Start-Window -command "`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8212"
    Start-Window -command "`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8213"

    Start-Window -command "`$env:LOGICAL_EID='2'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8221"
    Start-Window -command "`$env:LOGICAL_EID='2'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8222"

    Start-Window -command "`$env:LOGICAL_EID='3'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8231"
    Start-Window -command "`$env:LOGICAL_EID='3'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8232"

    Write-Host ">>> Function windows started."
}

# --------------------------------------------
# Default experiment knobs
# --------------------------------------------
$env:HOTSPOT_DRIFT_EVERY="50"
$env:HOTSPOT_SPAN="1"
$env:HOT_PROB="0.85"
$env:WARM_PROB="0.10"
$env:KEEP_ALIVE_STEPS="5"

# --- Serverless cold-start realism knobs (used by controller.py) ---
$env:KEEP_ALIVE_MS="5000"          # warm retention window in real time (ms)
$env:EVICTION_BASE_PROB="0.02"     # base reclaim probability
$env:EVICTION_TAU_MS="20000"       # reclaim grows with idle (ms)
$env:KEEPALIVE_MUL_HOT="1.5"       # hot path keeps warm longer
$env:KEEPALIVE_MUL_COLD="0.7"      # cold path more likely to go cold
$env:KEEPALIVE_MUL_HTTP="1.0"
$env:LOCAL_OOM_PROB="0.05"         # local contention probability (was 0.95 in old code)
$env:TRAFFIC_SKEW_ENABLE="1"       # 1=enable hotspot drift; 0=use natural gating only

$env:GRAD_HOT_PROB="0.90"
$env:GRAD_COLD_PROB="0.90"

$env:AUTOSCALE_ENABLE="1"
$env:AUTOSCALE_QUEUE_TH_MS="30"
$env:AUTOSCALE_MAX_REPLICA="6"
$env:AUTOSCALE_COOLDOWN_STEPS="8"
$env:MAX_STEPS="1000"
$env:DEADLINE_WARMUP_STEPS="30"
$env:DEADLINE_PCTL="95"
$env:DEADLINE_SAFETY="1.10"
$env:DEADLINE_MIN_MS="200"
$env:INVOKE_RETRIES="20"
$env:HOT_COVERAGE="0.70"
$env:HOTSET_MIN="1"
$env:HOTSET_MAX="4"   # 你的 NUM_EXPERTS

# --------------------------------------------
# Menu
# --------------------------------------------
Write-Host "`n================================================"
Write-Host "   SELECT EXPERIMENT MODE"
Write-Host "================================================"
Write-Host " [1] Run Ours (My Method)"
Write-Host " [2] Run Ablation Variants"
Write-Host " [3] Run Baseline Comparisons"
Write-Host " [4] Run ALL Experiments"
Write-Host "================================================"

$choice = Read-Host "Enter your choice (1-4)"

# --------------------------------------------
# Start functions
# --------------------------------------------
Start-AllFunctions
Start-Sleep -Seconds 2

# --------------------------------------------
# Run controller
# --------------------------------------------
function Run-Controller {
    param([string]$etype, [string]$mode)
    if ($etype -eq "baseline") {
        $env:EXPERIMENT_TYPE="baseline"
        $env:BASELINE_MODE=$mode
        $env:METRICS_FILE="metrics_baseline_$mode.csv"
    } else {
        $env:EXPERIMENT_TYPE="ablation"
        $env:ABLATION_MODE=$mode
        $env:METRICS_FILE="metrics_$mode.csv"
    }
    Write-Host ">>> Running controller: type=$etype mode=$mode metrics=$env:METRICS_FILE"
    python controller.py
}

switch ($choice) {
    "1" { Run-Controller -etype "ablation" -mode "full" }
    "2" {
        $modes = @("static_compute")
        # "full","no_hotcold","sync_update","static_compute","no_nsga", "no_online", "no_heuristic"
        foreach ($m in $modes) { Run-Controller -etype "ablation" -mode $m }
    }
    "3" {
        $modes = @("round_robin","greedy","bsp","ssp","asp","random","static")
        foreach ($m in $modes) {
            if ($m -eq "ssp") { $env:SSP_LIMIT="4" }
            Run-Controller -etype "baseline" -mode $m
        }
    }
    "4" {
        $ab = @("full","no_hotcold","sync_update","static_compute")
        foreach ($m in $ab) { Run-Controller -etype "ablation" -mode $m }

        $bl = @("round_robin","greedy","bsp","ssp","asp","random","static")
        foreach ($m in $bl) {
            if ($m -eq "ssp") { $env:SSP_LIMIT="4" }
            Run-Controller -etype "baseline" -mode $m
        }
    }
    default { Write-Host "[ERROR] Invalid selection." }
}
