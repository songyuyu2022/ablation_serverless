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
# export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

export KEEP_ALIVE_MS="5000"
export EVICTION_BASE_PROB="0.02"
export EVICTION_TAU_MS="20000"

# 拉大压力
export EXP_MAX_CONCURRENCY="1"
export HOT_PROB="0.75"
export COLD_EXPERT_LOAD_MS="400"

# GPU/AMP（3090 推荐）
export DEVICE="${DEVICE:-cuda}"
export AMP_ENABLED="${AMP_ENABLED:-1}"
export AMP_DTYPE="${AMP_DTYPE:-fp16}"

# 防 OOM/更稳定：先保守，跑通后再调大
export MAX_INFLIGHT_MICROBATCH="${MAX_INFLIGHT_MICROBATCH:-1}"
export MAX_INFLIGHT_EXPERT="${MAX_INFLIGHT_EXPERT:-1}"

# 你若使用“本地 in-process”，强烈建议关闭张量序列化（避免 CPU 内存爆）
export SERIALIZE_TENSORS="${SERIALIZE_TENSORS:-0}"

if [[ -z "${MAX_STEPS:-}" ]]; then
  export MAX_STEPS="2000"
fi

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
