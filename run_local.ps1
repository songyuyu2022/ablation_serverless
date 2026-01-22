# ============================================
# run_local.ps1 - Headless Local Simulation
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
$env:CUDA_VISIBLE_DEVICES = ""      # CPU only for controller logic

# --------------------------------------------
# KEY CONFIG: Enable Local Compute Mode
# --------------------------------------------
# "0" = Do NOT use external HTTP uvicorn servers
$env:USE_HTTP_EXEC="0"
# "1" = Enable In-Process LocalExecutor
$env:LOCAL_COMPUTE="1"

# --------------------------------------------
# Simulation Knobs
# --------------------------------------------
$env:HOTSPOT_DRIFT_EVERY="50"
$env:HOT_PROB="0.85"
$env:WARM_PROB="0.10"

# Serverless realism
$env:KEEP_ALIVE_MS="5000"
$env:EVICTION_BASE_PROB="0.02"
$env:EVICTION_TAU_MS="20000"
$env:AUTOSCALE_ENABLE="1"
$env:AUTOSCALE_QUEUE_TH_MS="30"
$env:AUTOSCALE_MAX_REPLICA="100" # Allowed to scale high locally
$env:VM_COLD_START_MS="2000"

# Experiment Settings
$env:MAX_STEPS="1000"
$env:LOG_TRAIN_EVERY="10"

# --------------------------------------------
# Menu
# --------------------------------------------
Write-Host "`n================================================"
Write-Host "   HEADLESS SIMULATION MODE"
Write-Host "   (No external windows required)"
Write-Host "================================================"
Write-Host " [1] Run Ours (My Method)"
Write-Host " [2] Run Ablation Variants"
Write-Host " [3] Run Baseline Comparisons"
Write-Host " [4] Run ALL Experiments"
Write-Host "================================================"

$choice = Read-Host "Enter your choice (1-4)"

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
        $modes = @("static_compute", "no_nsga", "no_online")
        foreach ($m in $modes) { Run-Controller -etype "ablation" -mode $m }
    }
    "3" {
        $modes = @("round_robin","greedy","random")
        foreach ($m in $modes) { Run-Controller -etype "baseline" -mode $m }
    }
    "4" {
        $ab = @("full","no_hotcold")
        foreach ($m in $ab) { Run-Controller -etype "ablation" -mode $m }
        $bl = @("round_robin","greedy")
        foreach ($m in $bl) { Run-Controller -etype "baseline" -mode $m }
    }
    default { Write-Host "[ERROR] Invalid selection." }
}