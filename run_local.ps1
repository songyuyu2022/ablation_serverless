# ============================================
# run_ablation_all.ps1 — 自动化运行所有消融实验
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
# 1. 全局环境与目录准备
# --------------------------------------------
$env:COMM_SIM_DIR        = "comm_sim"
$env:INSTANCES_FILE      = "instances.json"
$env:FUNC_MAP_FILE       = "func_map.json"
$env:CUDA_VISIBLE_DEVICES = ""      # CPU only

if (-not (Test-Path "$ROOT\comm_sim"))      { New-Item "$ROOT\comm_sim" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\hot"))  { New-Item "$ROOT\comm_sim\hot" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\cold")) { New-Item "$ROOT\comm_sim\cold" -ItemType Directory | Out-Null }

# 定义启动新窗口的函数（用于后台服务）
function Start-Window {
    param ( [string]$command )
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd `"$ROOT`"; . .\.venv\Scripts\Activate.ps1; $command"
    )
}

# --------------------------------------------
# 2. 启动后台服务 (如果服务已在运行，请手动关闭旧窗口或跳过此步)
# --------------------------------------------
Write-Host ">>> Starting background services (Pre/Post/Experts)..."

# --- pre_fn (8001-8003) ---
Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8001"
Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8002"
Start-Window -command "`$env:TOP_K='2'; `$env:NUM_EXPERTS='4'; uvicorn pre_fn:app --host 127.0.0.1 --port 8003"

# --- post_fn (8101-8103) ---
Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8101"
Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8102"
Start-Window -command "`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256'; uvicorn post_fn:app --host 127.0.0.1 --port 8103"

# --- Experts 0-3 (Ports 8201...8232) ---
# Expert 0
Start-Window -command "`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8201"
Start-Window -command "`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8202"
Start-Window -command "`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8203"
# Expert 1
Start-Window -command "`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8211"
Start-Window -command "`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8212"
Start-Window -command "`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8213"
# Expert 2
Start-Window -command "`$env:LOGICAL_EID='2'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8221"
Start-Window -command "`$env:LOGICAL_EID='2'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8222"
# Expert 3
Start-Window -command "`$env:LOGICAL_EID='3'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8231"
Start-Window -command "`$env:LOGICAL_EID='3'; `$env:EMB_DIM='256'; uvicorn expert_app:app --host 127.0.0.1 --port 8232"

Write-Host ">>> Waiting 10 seconds for services to initialize..."
Start-Sleep -Seconds 10

# --------------------------------------------
# 3. 循环运行所有消融实验 (Full Ablation Loop)
# --------------------------------------------

# 定义所有需要运行的模式
$AblationModes = @(
    "full",            # 完整模式
    "no_hotcold",      # 无冷热识别
    "sync_update",     # 同步更新
    "heuristic_only",  # 仅启发式调度
    "predictor_only",  # 仅预测器调度
    "no_nsga2"         # 无 NSGA-II
)

# 设置 Controller 通用参数
$env:TOP_K='2'; $env:NUM_EXPERTS='4'
$env:VOCAB_SIZE='2000'; $env:EMB_DIM='256'
$env:BATCH_SIZE='8'; $env:BLOCK_SIZE='64'
$env:MAX_STEPS='2200'; $env:VAL_INTERVAL='100'; $env:LOG_TRAIN_EVERY='10'
$env:MICRO_BATCHES="4"; $env:PARALLEL_DEGREE="4"

# 动态/网络参数
$env:HOTSPOT_DRIFT_EVERY="20"; $env:HOTSPOT_SPAN="3"; $env:HOT_PROB="0.80"; $env:WARM_PROB="0.10"
$env:GRAD_HOT_PROB="0.85"; $env:GRAD_COLD_PROB="0.85"
$env:AUTOSCALE_ENABLE="1"; $env:AUTOSCALE_QUEUE_TH_MS="30"; $env:AUTOSCALE_MAX_REPLICA="6"; $env:AUTOSCALE_COOLDOWN_STEPS="8"
$env:DEADLINE_WARMUP_STEPS="30"; $env:DEADLINE_PCTL="95"; $env:DEADLINE_SAFETY="1.10"; $env:DEADLINE_MIN_MS="200"

Write-Host ">>> Starting Ablation Loop: $AblationModes"

foreach ($mode in $AblationModes) {
    Write-Host "--------------------------------------------------------"
    Write-Host ">>> RUNNING MODE: $mode"
    Write-Host "--------------------------------------------------------"

    # 1. 设置当前模式
    $env:ABLATION_MODE = $mode

    # 2. 阻塞运行 Controller (不使用 Start-Window，而是直接 python)
    #    注意：这里假设 controller.py 会产生 metrics.csv 和 dispatch_trace.jsonl
    python controller.py

    # 3. 备份数据 (避免下一次运行覆盖)
    $timestamp = Get-Date -Format "yyyyMMdd-HHmm"

    if (Test-Path "metrics.csv") {
        $newMetricName = "metrics_${mode}.csv"
        Write-Host ">>> Renaming metrics.csv -> $newMetricName"
        Move-Item "metrics.csv" $newMetricName -Force
    } else {
        Write-Host "!!! Warning: metrics.csv not found for mode $mode"
    }

    if (Test-Path "dispatch_trace.jsonl") {
        $newTraceName = "trace_${mode}.jsonl"
        Write-Host ">>> Renaming dispatch_trace.jsonl -> $newTraceName"
        Move-Item "dispatch_trace.jsonl" $newTraceName -Force
    }

    Write-Host ">>> Finished mode: $mode"
    Write-Host "--------------------------------------------------------`n"

    # 可选：休息几秒让系统冷却
    Start-Sleep -Seconds 5
}

Write-Host ">>> All ablation experiments finished!"