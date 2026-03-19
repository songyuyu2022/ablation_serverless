#!/usr/bin/env bash
set -euo pipefail

echo ">>> Activating virtual environment..."
# 兼容两种 venv 路径（Ubuntu 通常是 .venv/bin/activate 或 venv/bin/activate）
VENV_FOUND=0
for p in "./.venv/bin/activate" "./venv/bin/activate"; do
  if [[ -f "$p" ]]; then
    # shellcheck disable=SC1090
    source "$p"
    VENV_FOUND=1
    break
  fi
done
if [[ "$VENV_FOUND" -eq 0 ]]; then
  echo "[WARNING] Virtual environment not found. Assuming Python is in PATH."
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo ">>> Project root = $ROOT"

# ------------------------------------------------
# Global env
# ------------------------------------------------
export COMM_SIM_DIR="comm_sim"
export INSTANCES_FILE="instances.json"
export FUNC_MAP_FILE="func_map.json"

# 单卡 3090：只用一张卡（可按需指定 0）
export CUDA_VISIBLE_DEVICES="0"
#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export USE_HTTP_EXEC="0"
export LOCAL_COMPUTE="1"

export DEFAULT_NET_LATENCY_MS="50.0"
export USE_TRACE_CALIB="1"

# Local transient failures (serverless realism)
export LOCAL_OOM_PROB="0.01"   # main experiments: small but non-zero
export LOCAL_BUSY_PROB="0.00"  # usually keep 0 unless stress testing

# 重试退避（更像真实 serverless）
export RETRY_BACKOFF_MS="5"
export RETRY_BACKOFF_MAX_MS="50"

# Network latency sanity protection
export MIN_NET_LATENCY_RATIO="0.05"

export KEEP_ALIVE_MS="1000"
export EVICTION_BASE_PROB="0.09"
export EVICTION_TAU_MS="3000"

# 拉大压力
export FAST_LOCAL_BYPASS_COMM="1"
export SEQ_LEN=512
export MICRO_BATCH=8
export BATCH_SIZE=16
export MAX_INFLIGHT_MICROBATCH="2"
export MAX_INFLIGHT_EXPERT="4"
export EXP_MAX_CONCURRENCY="4"
export HOT_PROB="0.40"
export NUM_EXPERTS="16"          # 极其关键！如果不加，默认只有 4 个专家，冷启动出不来
export WARM_PROB="0.25"          # 配合 HOT_PROB 打散流量
export HOTSPOT_DRIFT_EVERY="20"  # 让热点频繁漂移，制造真实的系统动荡
export COLD_EXPERT_LOAD_MS="400"
export EMB_DIM=768          # 512 -> 768（模型更重
export TOP_K=2              # 保持2，别太大
export AMP_DTYPE=bf16
export GRAD_CLIP_NORM=1.0

# GPU/AMP（3090 推荐）
export DEVICE="${DEVICE:-cuda}"
export AMP_ENABLED="${AMP_ENABLED:-1}"

# 你若使用“本地 in-process”，强烈建议关闭张量序列化（避免 CPU 内存爆）
export SERIALIZE_TENSORS="${SERIALIZE_TENSORS:-0}"

# -----------------------------
# Hot/Cold hysteresis tuning (让冷热差异更明显)
# -----------------------------
export HOTSET_SIZE="2"            # 先从 1 开始最容易拉开差异
export HOT_ENTER_P="0.55"
export HOT_EXIT_P="0.45"
export HOT_MIN_STAY_STEPS="8"
export HEATMAP_UPDATE_EVERY="1"
export HEATMAP_DECAY="0.90"
export HEATMAP_TREND_DECAY="0.85"

# -----------------------------
# Acc logging (画图友好：未计算写 NaN)
# -----------------------------
export ACC_EVERY="1"              # 每 5 step 计算一次 acc5（可调 1/5/10）
export ACC_FILL_NAN="0"

# -----------------------------
# Predictor monitoring target (让 R2 更稳定)
# -----------------------------
export PRED_TARGET="nocold"       # 监控时用 (actual - cold) 对齐预测更稳
export PRED_MONITOR_WINDOW="200"  # 增大 R2 统计窗口（默认 50 太短）
export MAX_STEPS="500"
export MAX_GPU_COMPUTE_MS="800.0"  # 把上限放宽到 800ms
export COLD_ACC_STEPS="5"    # 让冷专家积攒 10 步（约 5 秒）才去更新

export LR_PRE="3e-4"
export LR_POST="3e-4"
export LR_EXP="3e-4"

export NN_TRUST_WARMUP="60"
export NN_TRUST_R2_TH="0.20"
export NN_TRUST_MAE_TH="200.0"
export NN_CLIP_LO="0.6"
export NN_CLIP_HI="1.6"

# ------------------------------------------------
# Result root directory (NEW)
# ------------------------------------------------
RESULT_ROOT="$ROOT/results"
mkdir -p "$RESULT_ROOT"

TS="$(date +%F_%H-%M-%S)"
RUN_DIR="$RESULT_ROOT/$TS"
mkdir -p "$RUN_DIR"

echo ">>> Results will be saved to:"
echo "    $RUN_DIR"

# ------------------------------------------------
# Menu
# ------------------------------------------------
echo "================================================"
echo " Serverless MoE Local Simulation"
echo " Multi-Seed + Result Directory"
echo "================================================"
echo " [1] Run Ours (Full)"
echo " [2] Run Ablations"
echo " [3] Run Baselines"
echo " [4] Run ALL (Paper Set)"
echo " [5] Run ONLY missing modes (BSP/SSP/ASP + no_heuristic)"
echo " [6] Run Custom (自由输入想跑的单个或多个模式)"  # <--- 新增这行
echo "================================================"
echo " [S] Single seed"
echo " [M] Multi-seed (0,1,2)"
echo "================================================"

read -r -p "Enter choice (e.g. 4 M): " choice
# 拆分输入：例如 "4 M"
read -r exp_choice seed_mode <<<"$choice"
seed_mode="${seed_mode:-M}"
seed_mode="$(echo "$seed_mode" | tr '[:lower:]' '[:upper:]')"

# Seeds
if [[ "$seed_mode" == "S" ]]; then
  read -r -p "Enter SEED (default 0): " seed_in
  seed_in="${seed_in:-0}"
  SEEDS=("$seed_in")
else
  SEEDS=(0 1 2)
fi

# ------------------------------------------------
# Cleanup
# ------------------------------------------------
cleanup_data() {
  if [[ -d "comm_sim" ]]; then
    rm -rf "comm_sim" || true
  fi
  # 清理 __pycache__
  find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
  sleep 0.2
}

# ------------------------------------------------
# Run controller
# ------------------------------------------------
run_controller() {
  local etype="$1"
  local mode="$2"
  local seed="$3"

  export SEED="$seed"
  cleanup_data

  local subdir="$RUN_DIR/$etype"
  mkdir -p "$subdir"

  if [[ "$etype" == "baseline" ]]; then
    export EXPERIMENT_TYPE="baseline"
    export BASELINE_MODE="$mode"
    export METRICS_FILE="$subdir/metrics_baseline_${mode}_seed${seed}.csv"
  else
    export EXPERIMENT_TYPE="ablation"
    export ABLATION_MODE="$mode"
    export METRICS_FILE="$subdir/metrics_${mode}_seed${seed}.csv"
  fi

  echo ">>> Running $etype / $mode / seed=$seed"
  echo ">>> Output: $METRICS_FILE"

  # 同时把 stdout/stderr 存日志（方便云上排查）
  python controller.py 2>&1 | tee "$subdir/log_${etype}_${mode}_seed${seed}.txt"

  echo ">>> Finished $etype / $mode / seed=$seed"
  echo "------------------------------------------------"
}

# ------------------------------------------------
# Dispatch
# ------------------------------------------------
run_choice() {
  local c="$1"
  local seed="$2"

  case "$c" in
    "1")
      run_controller "ablation" "full" "$seed"
      ;;
    "2")
      for m in no_nsga no_online no_hotcold; do
        run_controller "ablation" "$m" "$seed"
      done
      ;;
    "3")
      for m in round_robin random; do
        run_controller "baseline" "$m" "$seed"
      done
      ;;
    "4")
      run_controller "ablation" "full" "$seed"
      for m in round_robin random; do
        run_controller "baseline" "$m" "$seed"
      done
      for m in no_hotcold no_nsga no_online; do
        run_controller "ablation" "$m" "$seed"
      done
      ;;
    "5")
      # Only run missing ones:
      # baselines: bsp / asp
      for m in bsp asp; do
        run_controller "baseline" "$m" "$seed"
      done
      # ablation: no_heuristic
      run_controller "ablation" "no_heuristic" "$seed"
      ;;
    "6")
      # ==========================================================
      # 👇 在这里直接修改你想跑的模式（用空格隔开）👇
      # 可选：round_robin random bsp ssp asp greedy no_nsga no_online no_hotcold no_heuristic
      # ==========================================================
      local custom_modes="no_nsga no_online full no_heuristic"

      for m in $custom_modes; do
        if [[ "$m" == "round_robin" || "$m" == "random" || "$m" == "asp" || "$m" == "bsp" || "$m" == "ssp" || "$m" == "greedy" ]]; then
          run_controller "baseline" "$m" "$seed"
        else
          run_controller "ablation" "$m" "$seed"
        fi
      done
      ;;
    *)
      echo "Invalid selection."
      ;;
  esac
}

for s in "${SEEDS[@]}"; do
  echo "=============================="
  echo " Running seed = $s"
  echo "=============================="
  run_choice "$exp_choice" "$s"
done

echo "All runs finished."
echo "Results saved under:"
echo "  $RUN_DIR"
