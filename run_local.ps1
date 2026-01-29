# ============================================
# run_local.ps1 - Headless Local Simulation
# ============================================

Write-Host ">>> Activating virtual environment..."
# 检查多种可能的 venv 路径，增强兼容性
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
    Write-Host "[WARNING] Virtual environment not found in standard paths. Assuming Python is in PATH."
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
# "1" = Enable In-Process LocalExecutor (Crucial for your new architecture)
$env:LOCAL_COMPUTE="1"

# --------------------------------------------
# Simulation Knobs
# --------------------------------------------
$env:HOTSPOT_DRIFT_EVERY="50"
$env:HOT_PROB="0.85"
$env:WARM_PROB="0.10"
$env:LOCAL_OOM_PROB="0"

# Serverless realism
$env:KEEP_ALIVE_MS="5000"
$env:EVICTION_BASE_PROB="0.02"
$env:EVICTION_TAU_MS="20000"

Write-Host "================================================"
Write-Host "    Serverless MoE Local Simulation (No-GUI)    "
Write-Host "    Mode: LocalExecutor (In-Process)            "
Write-Host "    Storage: File-based CommManager (comm_sim)  "
Write-Host "================================================"
Write-Host " [1] Run Ours (My Method)"
Write-Host " [2] Run Ablation Variants (Static, No-NSGA, etc.)"
Write-Host " [3] Run Baseline Comparisons (RR, Greedy, Random)"
Write-Host " [4] Run ALL Experiments"
Write-Host "================================================"

$choice = Read-Host "Enter your choice (1-4)"

# [新增] 清理函数：防止上一次实验的残留文件影响本次实验
function Cleanup-Data {
    # 1. [核心] 清理仿真存储 (必须)
    if (Test-Path "comm_sim") {
        Write-Host ">>> [Cleanup] Removing storage directory (comm_sim)..." -ForegroundColor Yellow
        Remove-Item -Path "comm_sim" -Recurse -Force -ErrorAction SilentlyContinue
    }

    # 2. [可选] 清理 Python 缓存 (推荐清理，防止代码修改不生效)
    Get-ChildItem -Path . -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # 3. [可选] 清理词表缓存 (仅在修改 input.txt 后需要打开)
    # if (Test-Path "vocab.json") { Remove-Item "vocab.json" }

    # 等待文件锁释放
    Start-Sleep -Milliseconds 200
}

function Run-Controller {
    param([string]$etype, [string]$mode)

    # 每次运行前清理数据
    Cleanup-Data

    if ($etype -eq "baseline") {
        $env:EXPERIMENT_TYPE="baseline"
        $env:BASELINE_MODE=$mode
        $env:METRICS_FILE="metrics_baseline_$mode.csv"
    } else {
        $env:EXPERIMENT_TYPE="ablation"
        $env:ABLATION_MODE=$mode
        $env:METRICS_FILE="metrics_$mode.csv"
    }

    # 如果目标 CSV 存在，先删除，保证数据是新的
    if (Test-Path $env:METRICS_FILE) {
        Remove-Item $env:METRICS_FILE
    }

    Write-Host ">>> Running controller: type=$etype mode=$mode output=$env:METRICS_FILE" -ForegroundColor Green
    python controller.py

    Write-Host ">>> Finished $mode. Data saved to $env:METRICS_FILE" -ForegroundColor Cyan
    Write-Host "------------------------------------------------"
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
        # Ours
        Run-Controller -etype "ablation" -mode "full"
        # Baselines
        $bl = @("round_robin","greedy","random")
        foreach ($m in $bl) { Run-Controller -etype "baseline" -mode $m }
        # Ablations
        $ab = @("static_compute", "no_nsga", "no_online")
        foreach ($m in $ab) { Run-Controller -etype "ablation" -mode $m }
    }
    Default { Write-Host "Invalid selection." }
}