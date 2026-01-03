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
# 1. Global Setup
# --------------------------------------------
$env:COMM_SIM_DIR        = "comm_sim"
$env:INSTANCES_FILE      = "instances.json"
$env:FUNC_MAP_FILE       = "func_map.json"
$env:CUDA_VISIBLE_DEVICES = ""      # CPU only

if (-not (Test-Path "$ROOT\comm_sim"))      { New-Item "$ROOT\comm_sim" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\hot"))  { New-Item "$ROOT\comm_sim\hot" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\cold")) { New-Item "$ROOT\comm_sim\cold" -ItemType Directory | Out-Null }

# Helper function to start background windows
function Start-Window {
    param ( [string]$command )
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd `"$ROOT`"; . .\.venv\Scripts\Activate.ps1; $command"
    )
}

# --------------------------------------------
# 2. Interactive Selection
# --------------------------------------------
Write-Host "`n================================================"
Write-Host "   SELECT EXPERIMENT MODE"
Write-Host "================================================"
Write-Host " [1] Run Ours (My Method)"
Write-Host "     Runs only 'full' mode for debugging."
Write-Host ""
Write-Host " [2] Run Ablation Variants"
Write-Host "     Runs 'no_hotcold', 'sync_update' etc."
Write-Host ""
Write-Host " [3] Run Baseline Comparisons"
Write-Host "     Runs 'round_robin', 'static', 'bsp' etc."
Write-Host ""
Write-Host " [4] Run ALL Experiments"
Write-Host "     Sequence: Ours -> Ablation -> Baseline"
Write-Host "================================================"

$selection = Read-Host "Enter number (1/2/3/4) [Default: 1]"
if ($selection -eq "") { $selection = "1" }

$RunOurs = $false
$RunAblation = $false
$RunBaseline = $false

switch ($selection) {
    "1" {
        $RunOurs = $true
        Write-Host ">>> Selected: [Run Ours]" -ForegroundColor Cyan
    }
    "2" {
        $RunAblation = $true
        Write-Host ">>> Selected: [Run Ablation Variants]" -ForegroundColor Cyan
    }
    "3" {
        $RunBaseline = $true
        Write-Host ">>> Selected: [Run Baseline Comparisons]" -ForegroundColor Cyan
    }
    "4" {
        $RunOurs = $true; $RunAblation = $true; $RunBaseline = $true
        Write-Host ">>> Selected: [Run ALL]" -ForegroundColor Cyan
    }
    Default {
        $RunOurs = $true
        Write-Host ">>> Invalid input. Defaulting to: [Run Ours]" -ForegroundColor Yellow
    }
}

# --------------------------------------------
# 3. Start Background Services
# --------------------------------------------
$startServices = Read-Host "`nStart background services (pre/post/experts)? (y/n) [Default: n]"
if ($startServices -eq "y") {
    Write-Host ">>> Starting background services..."

    # --- pre_fn (8001-8003) ---
    Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8001"
    Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8002"
    Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8003"

    # --- post_fn (8101-8103) ---
    Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8101"
    Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8102"
    Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8103"

    # --- Experts 0-3 (Ports 8201...8232) ---
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

    Write-Host ">>> Waiting 10 seconds for services to initialize..."
    Start-Sleep -Seconds 10
}

# --------------------------------------------
# 4. Common Configuration
# --------------------------------------------
$env:TOP_K='2'; $env:NUM_EXPERTS='4'
$env:VOCAB_SIZE='2000'; $env:EMB_DIM='256'
$env:BATCH_SIZE='8'; $env:BLOCK_SIZE='64'
$env:MAX_STEPS='800'; $env:VAL_INTERVAL='100'; $env:LOG_TRAIN_EVERY='10'
$env:MICRO_BATCHES="4"; $env:PARALLEL_DEGREE="4"

$env:HOTSPOT_DRIFT_EVERY="100"
$env:HOTSPOT_SPAN="1"
$env:HOT_PROB="0.85"
$env:WARM_PROB="0.10"
$env:KEEP_ALIVE_STEPS="5"

$env:GRAD_HOT_PROB="0.90"
$env:GRAD_COLD_PROB="0.90"

$env:AUTOSCALE_ENABLE="1"; $env:AUTOSCALE_QUEUE_TH_MS="30"; $env:AUTOSCALE_MAX_REPLICA="6"; $env:AUTOSCALE_COOLDOWN_STEPS="8"
$env:DEADLINE_WARMUP_STEPS="30"; $env:DEADLINE_PCTL="95"; $env:DEADLINE_SAFETY="1.10"; $env:DEADLINE_MIN_MS="200";$env:INVOKE_RETRIES="20"

# --------------------------------------------
# 5. [Mode] Ours
# --------------------------------------------
if ($RunOurs) {
    Write-Host "`n>>> Starting [Ours: My Method]..." -ForegroundColor Green

    $mode = "full"
    Write-Host "--------------------------------------------------------"
    Write-Host ">>> [Ours] RUNNING MODE: $mode"
    Write-Host "--------------------------------------------------------"

    $env:EXPERIMENT_TYPE = "ablation"
    $env:ABLATION_MODE = $mode

    Remove-Item Env:\BASELINE_MODE -ErrorAction SilentlyContinue

    python controller.py

    if (Test-Path "metrics.csv") {
        $newMetricName = "metrics_${mode}.csv"
        Write-Host ">>> Renaming metrics.csv -> $newMetricName"
        Move-Item "metrics.csv" $newMetricName -Force
    }
    if (Test-Path "dispatch_trace.jsonl") {
        $newTraceName = "trace_${mode}.jsonl"
        Write-Host ">>> Renaming dispatch_trace.jsonl -> $newTraceName"
        Move-Item "dispatch_trace.jsonl" $newTraceName -Force
    }

    Start-Sleep -Seconds 3
}

# --------------------------------------------
# 6. [Mode] Ablation Variants
# --------------------------------------------
if ($RunAblation) {
    $AblationModes = @(
        "no_hotcold",
        "sync_update",
        "heuristic_only",
        "predictor_only",
        "no_nsga2"
    )

    Write-Host "`n>>> Starting [Ablation Variants] Loop..." -ForegroundColor Green

    foreach ($mode in $AblationModes) {
        Write-Host "--------------------------------------------------------"
        Write-Host ">>> [Ablation] RUNNING MODE: $mode"
        Write-Host "--------------------------------------------------------"

        $env:EXPERIMENT_TYPE = "ablation"
        $env:ABLATION_MODE = $mode

        Remove-Item Env:\BASELINE_MODE -ErrorAction SilentlyContinue

        python controller.py

        if (Test-Path "metrics.csv") {
            $newMetricName = "metrics_${mode}.csv"
            Write-Host ">>> Renaming metrics.csv -> $newMetricName"
            Move-Item "metrics.csv" $newMetricName -Force
        }
        if (Test-Path "dispatch_trace.jsonl") {
            $newTraceName = "trace_${mode}.jsonl"
            Write-Host ">>> Renaming dispatch_trace.jsonl -> $newTraceName"
            Move-Item "dispatch_trace.jsonl" $newTraceName -Force
        }

        Start-Sleep -Seconds 3
    }
}

# --------------------------------------------
# 7. [Mode] Baseline Comparisons
# --------------------------------------------
if ($RunBaseline) {
    $BaselineModes = @(
        "round_robin",
        "greedy",
        "static",
        "bsp",
        "asp",
        "ssp"
    )

    Write-Host "`n>>> Starting [Baseline] Loop..." -ForegroundColor Green

    foreach ($mode in $BaselineModes) {
        Write-Host "--------------------------------------------------------"
        Write-Host ">>> [Baseline] RUNNING MODE: $mode"
        Write-Host "--------------------------------------------------------"

        $env:EXPERIMENT_TYPE = "baseline"
        $env:BASELINE_MODE = $mode

        Remove-Item Env:\ABLATION_MODE -ErrorAction SilentlyContinue

        if ($mode -eq "ssp") {
            $env:SSP_LIMIT = "4"
        }

        python controller.py

        if (Test-Path "metrics.csv") {
            $newMetricName = "metrics_baseline_${mode}.csv"
            Write-Host ">>> Renaming metrics.csv -> $newMetricName"
            Move-Item "metrics.csv" $newMetricName -Force
        }
        if (Test-Path "dispatch_trace.jsonl") {
            $newTraceName = "trace_baseline_${mode}.jsonl"
            Write-Host ">>> Renaming dispatch_trace.jsonl -> $newTraceName"
            Move-Item "dispatch_trace.jsonl" $newTraceName -Force
        }

        Start-Sleep -Seconds 3
    }
}

Write-Host ""
Write-Host "Done."