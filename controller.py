# -------------------------------------------------------------------------
# [UPDATED - FULL REPLACE controller.py]
# 新增特性：Local Compute Mode (无窗口计算模式)
# 1) 引入 LocalExecutor，在 controller 进程内直接运行 Pre/Post/Expert 模型
# 2) 拦截 USE_HTTP_EXEC=0 的情况，转为调用 LocalExecutor
# 3) 完美保留所有延迟仿真 (Autoscaler, Azure Trace, Cold Start)
# -------------------------------------------------------------------------
# [NEW] 引入新的指标记录器
from utils.metrics import StepMetrics, MetricsLogger
# from __future__ import annotations
import os
import asyncio
import json
import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from comm import CommManager
# ✅ 使用你项目既有的序列化协议
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
# 引入适配器
from makemoe_adapter import MakeMoEAdapter
from makeMoE import MakeMoEConfig
from comm import CommManager
import requests
from collections import deque

# ============================================================
# Tail / stability metrics (rolling + global) and concurrency limiters
# - rolling: recent N steps (for plots)
# - global : approximate p95/p99 via bounded reservoir + exact CV via Welford (for tables)
# ============================================================
TAIL_STAT_WINDOW = int(os.getenv("TAIL_STAT_WINDOW", "200"))
TAIL_STAT_MIN_SAMPLES = int(os.getenv("TAIL_STAT_MIN_SAMPLES", "5"))
GLOBAL_STAT_RESERVOIR = int(os.getenv("GLOBAL_STAT_RESERVOIR", "5000"))

# Rolling window per phase
_STEP_TIME_ROLL: Dict[str, deque] = {"train": deque(maxlen=TAIL_STAT_WINDOW), "val": deque(maxlen=TAIL_STAT_WINDOW)}
# Bounded reservoir samples for global percentiles
_STEP_TIME_RES: Dict[str, List[float]] = {"train": [], "val": []}
# Welford running stats for global CV (exact mean/std without storing all samples)
_STEP_TIME_WELFORD: Dict[str, Dict[str, float]] = {
    "train": {"n": 0.0, "mean": 0.0, "m2": 0.0},
    "val": {"n": 0.0, "mean": 0.0, "m2": 0.0},
}


def _tail_stats_from_list(xs: List[float]) -> Tuple[float, float, float]:
    """Return (p95, p99, cv) for a list of step_time_ms samples."""
    n = len(xs)
    if n < TAIL_STAT_MIN_SAMPLES:
        return 0.0, 0.0, 0.0
    arr = np.asarray(xs, dtype=np.float64)
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    mu = float(arr.mean())
    if mu <= 1e-12 or n < 2:
        cv = 0.0
    else:
        cv = float(arr.std(ddof=0) / mu)
    return p95, p99, cv


def _welford_update(state: Dict[str, float], x: float) -> None:
    n = state["n"] + 1.0
    delta = x - state["mean"]
    mean = state["mean"] + delta / n
    delta2 = x - mean
    m2 = state["m2"] + delta * delta2
    state["n"] = n
    state["mean"] = mean
    state["m2"] = m2


def _welford_cv(state: Dict[str, float]) -> float:
    n = state["n"]
    mean = state["mean"]
    if n < 2 or abs(mean) <= 1e-12:
        return 0.0
    var = state["m2"] / n
    std = math.sqrt(max(0.0, var))
    return float(std / mean)


def _reservoir_add(res: List[float], x: float, *, cap: int, seen: int) -> None:
    """Reservoir sampling: keep at most cap samples, unbiased over stream."""
    if cap <= 0:
        return
    if len(res) < cap:
        res.append(x)
        return
    j = random.randint(0, max(0, seen - 1))
    if j < cap:
        res[j] = x


# Concurrency limiters
MAX_INFLIGHT_MICROBATCH = int(os.getenv("MAX_INFLIGHT_MICROBATCH", "1"))
MAX_INFLIGHT_EXPERT = int(os.getenv("MAX_INFLIGHT_EXPERT", "1"))
_MB_SEM = asyncio.Semaphore(MAX_INFLIGHT_MICROBATCH)
_EXPERT_SEM = asyncio.Semaphore(MAX_INFLIGHT_EXPERT)


async def _limited_mb(coro):
    async with _MB_SEM:
        return await coro


async def _limited_expert(coro):
    async with _EXPERT_SEM:
        return await coro


# [Insert in controller.py] 辅助类：计算预测器 R2 Score
class PredictionMonitor:
    """
    Online prediction monitor.
    - MAE(t) = | pred(t-1) - actual(t) |
    - R2 computed on lagged pairs, with variance guard
    """

    def __init__(self, window: int = 50):
        self.window = window

        # lagged prediction buffer
        self._last_pred = None

        # store error pairs
        self._y_true = deque(maxlen=window)
        self._y_pred = deque(maxlen=window)

    def update(self, *, pred_lat: float, actual_lat: float):
        """
        Update monitor with current prediction and actual latency.
        """
        # record MAE / R2 using last prediction
        if self._last_pred is not None and actual_lat is not None:
            try:
                y_t = float(actual_lat)
                y_hat = float(self._last_pred)

                # allow zero / small values, just ignore NaN / inf
                if np.isfinite(y_t) and np.isfinite(y_hat):
                    self._y_true.append(y_t)
                    self._y_pred.append(y_hat)
            except Exception:
                pass

        # update last prediction for next step
        if pred_lat is not None:
            try:
                p = float(pred_lat)
                if np.isfinite(p):
                    self._last_pred = p
            except Exception:
                pass

    def get_mae(self):
        if len(self._y_true) == 0:
            return 0.0
        y_t = np.asarray(self._y_true)
        y_p = np.asarray(self._y_pred)
        return float(np.mean(np.abs(y_p - y_t)))

    def get_r2(self):
        """
        Numerically stable R2.
        Returns 0.0 when variance too small or samples insufficient.
        """
        n = len(self._y_true)
        if n < 3:
            return 0.0

        y_t = np.asarray(self._y_true)
        y_p = np.asarray(self._y_pred)

        # total sum of squares
        var = np.var(y_t)
        if var < 1e-6:
            return 0.0

        sse = np.sum((y_p - y_t) ** 2)
        sst = np.sum((y_t - np.mean(y_t)) ** 2)

        if sst <= 1e-9:
            return 0.0

        r2 = 1.0 - sse / sst

        # clip to reasonable range (avoid extreme negatives)
        return float(np.clip(r2, -1.0, 1.0))

    def get_r2_mae(self):
        """
        Compatibility helper if you were calling get_r2_mae()
        """
        return self.get_r2(), self.get_mae()


# 实例化全局监控器
PRED_MONITOR = PredictionMonitor(window=int(os.getenv("PRED_MONITOR_WINDOW", "50")))

class RealDataLoader:
    def __init__(self, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.data = self._load_data()
        self.vocab_size = 0  # 会在 prepare 后更新
        self.stoi = {}
        self.itos = {}
        self.train_data = None
        self.val_data = None
        self._prepare()

    def _load_data(self):
        file_path = 'input.txt'
        if not os.path.exists(file_path):
            print(">>> [Data] Downloading input.txt (Tiny Shakespeare)...")
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _prepare(self):
        chars = sorted(list(set(self.data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # 编码整个数据集
        tensor_data = torch.tensor([self.stoi[c] for c in self.data], dtype=torch.long)
        n = int(0.9 * len(tensor_data))
        self.train_data = tensor_data[:n]
        self.val_data = tensor_data[n:]
        print(f">>> [Data] Loaded {len(self.data)} chars, vocab_size={self.vocab_size}")

    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x, y


# ============================================================
# Global Env
# ============================================================
LOCAL_EXECUTOR = None
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

COMM_SIM_DIR = os.getenv("COMM_SIM_DIR", "comm_sim")
INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")

EXPERIMENT_TYPE = os.getenv("EXPERIMENT_TYPE", "ablation")
ABLATION_MODE = os.getenv("ABLATION_MODE", "full")
BASELINE_MODE = os.getenv("BASELINE_MODE", "round_robin")

# Training
MAX_STEPS = int(os.getenv("MAX_STEPS", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MICRO_BATCH = int(os.getenv("MICRO_BATCH", "4"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "20"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))

# Sequence / token LM batch
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))
DATA_PATH = os.getenv("DATA_PATH", "input.txt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")

# MoE
NUM_EXPERTS = int(os.getenv("NUM_EXPERTS", "16"))
TOP_K = int(os.getenv("TOP_K", "2"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "2000"))
EMB_DIM = int(os.getenv("EMB_DIM", "512"))

# Hot/Cold logic
HOTSET_SIZE = int(os.getenv("HOTSET_SIZE", "1"))
HEATMAP_DECAY = float(os.getenv("HEATMAP_DECAY", "0.98"))
HEATMAP_MIN_PROB = float(os.getenv("HEATMAP_MIN_PROB", "0.01"))

# Traffic skew
HOTSPOT_DRIFT_EVERY = int(os.getenv("HOTSPOT_DRIFT_EVERY", "50"))
HOTSPOT_SPAN = int(os.getenv("HOTSPOT_SPAN", "1"))
HOT_PROB = float(os.getenv("HOT_PROB", "0.85"))
WARM_PROB = float(os.getenv("WARM_PROB", "0.15"))
TRAFFIC_SKEW_ENABLE = os.getenv("TRAFFIC_SKEW_ENABLE", "1") == "1"

# Network multipliers
DEFAULT_NET_LATENCY = float(os.getenv("DEFAULT_NET_LATENCY_MS", "50.0"))
DEFAULT_PERFORMANCE = float(os.getenv("DEFAULT_PERFORMANCE", "1.0"))
HOT_NET_MUL = float(os.getenv("HOT_NET_MUL", "0.5"))
COLD_NET_MUL = float(os.getenv("COLD_NET_MUL", "2.0"))
HTTP_NET_MUL = float(os.getenv("HTTP_NET_MUL", "1.0"))
SHARED_NET_MUL = float(os.getenv("SHARED_NET_MUL", "0.2"))  # 内存级高速通道，默认更快
FALLBACK_NET_MUL = float(os.getenv("FALLBACK_NET_MUL", "1.3"))
COLD_STORAGE_MS = float(os.getenv("COLD_STORAGE_MS", "12.0"))

# Retry & SLO
INVOKE_RETRIES = int(os.getenv("INVOKE_RETRIES", "10"))
DEADLINE_WARMUP_STEPS = int(os.getenv("DEADLINE_WARMUP_STEPS", "30"))
DEADLINE_PCTL = int(os.getenv("DEADLINE_PCTL", "95"))
DEADLINE_SAFETY = float(os.getenv("DEADLINE_SAFETY", "1.5"))
DEADLINE_MIN_MS = float(os.getenv("DEADLINE_MIN_MS", "800"))

# ---- HTTP / LOCAL EXECUTION SWITCH ----
# "1" = Real HTTP requests (needs uvicorn), "0" = Local In-Process Simulation
USE_HTTP_EXEC = os.getenv("USE_HTTP_EXEC", "1") == "1"
# Force local compute if explicitly set
LOCAL_COMPUTE = os.getenv("LOCAL_COMPUTE", "0") == "1"

HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "30"))
SIM_SLEEP = os.getenv("SIM_SLEEP", "0") == "1"
HTTP_CONCURRENCY = int(os.getenv("HTTP_CONCURRENCY", "32"))

# endpoint paths
PATH_FWD = os.getenv("PATH_FWD", "/fwd")
PATH_BWD = os.getenv("PATH_BWD", "/bwd")
PATH_STEP = os.getenv("PATH_STEP", "/step")
PATH_ZERO = os.getenv("PATH_ZERO", "/zero")
PATH_HEALTH = os.getenv("PATH_HEALTH", "/health")

# Async backward policy
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "4"))
FORCE_SYNC_UPDATE = os.getenv("FORCE_SYNC_UPDATE", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Autoscaling Configuration
# ============================================================
AUTOSCALE_ENABLE = os.getenv("AUTOSCALE_ENABLE", "1") == "1"
AUTOSCALE_QUEUE_TH_MS = float(os.getenv("AUTOSCALE_QUEUE_TH_MS", "50.0"))
AUTOSCALE_IDLE_TIMEOUT_S = float(os.getenv("AUTOSCALE_IDLE_TIMEOUT_S", "600.0"))
AUTOSCALE_MIN_REPLICAS = int(os.getenv("AUTOSCALE_MIN_REPLICAS", "1"))
VM_COLD_START_MS = float(os.getenv("VM_COLD_START_MS", "2000.0"))


# ============================================================
# Ablation config
# ============================================================

@dataclass
class AblationConfig:
    is_baseline: bool = False
    disable_hotcold: bool = False
    force_sync_update: bool = False
    use_random_sched: bool = False
    use_rr_sched: bool = False
    use_bsp: bool = False
    use_ssp: bool = False
    use_asp: bool = False
    disable_nsga: bool = False
    disable_online_pred: bool = False
    disable_heuristic: bool = False

    @staticmethod
    def from_env() -> "AblationConfig":
        cfg = AblationConfig()
        if EXPERIMENT_TYPE == "baseline":
            cfg.is_baseline = True
            m = BASELINE_MODE.lower()
            cfg.use_rr_sched = (m == "round_robin")
            cfg.use_random_sched = (m == "random")
            cfg.use_bsp = (m == "bsp")
            cfg.use_ssp = (m == "ssp")
            cfg.use_asp = (m == "asp")
            if m in ["random", "round_robin"]:
                cfg.disable_nsga = True  # 关掉统筹算法
                cfg.disable_heuristic = True  # 关掉聪明打分公式
            elif m == "greedy":
                cfg.disable_nsga = True  # Greedy 只关掉统筹，保留打分
        else:
            m = ABLATION_MODE.lower()
            if m == "no_hotcold":
                cfg.disable_hotcold = True
            elif m == "sync_update":
                cfg.force_sync_update = True
            elif m == "random_sched":
                cfg.use_random_sched = True
            elif m == "rr_sched":
                cfg.use_rr_sched = True
            elif m == "no_nsga":
                cfg.disable_nsga = True
            elif m == "no_online":
                cfg.disable_online_pred = True
            elif m == "no_heuristic":
                cfg.disable_heuristic = True
        return cfg


ABL_CFG = AblationConfig.from_env()
if FORCE_SYNC_UPDATE:
    ABL_CFG.force_sync_update = True


# ============================================================
# NEW: Local Executor (In-Process Calculation)
# ============================================================
# controller.py 中的 LocalExecutor 类
class LocalExecutor:
    """
    Local in-process executor for serverless MoE pipeline (v2).

    v2 improvements:
    - When FAST_LOCAL_BYPASS_COMM=1 and SERIALIZE_TENSORS=0:
      * NO tensor_to_pack in the hot path
      * pass torch.Tensor directly between stages
      This avoids GPU->CPU sync/copies and typically boosts GPU utilization a lot.
    - Optional AMP autocast + GradScaler (controlled by AMP_ENABLED/AMP_DTYPE)
    - Fine-grained locks only (no global lock)
    """

    def __init__(self):
        self.device = torch.device(DEVICE)
        print(f"[LocalExecutor] Initializing MakeMoE Adapter on {self.device}...")

        # knobs
        self.fast_bypass_comm = os.getenv("FAST_LOCAL_BYPASS_COMM", "1") == "1"
        self.serialize_tensors = os.getenv("SERIALIZE_TENSORS", "0") == "1"
        # direct tensor IO is the key for GPU util in local compute
        self.direct_tensor_io = self.fast_bypass_comm and (not self.serialize_tensors)

        # acc 计算频率：优先读 ACC_EVERY，其次 LOCAL_ACC_EVERY
        self.compute_acc_every = int(os.getenv("ACC_EVERY", os.getenv("LOCAL_ACC_EVERY", "10")))
        self.acc_fill_nan = (os.getenv("ACC_FILL_NAN", "0") == "1")
        self._acc_counter = 0

        # AMP
        # AMP
        self.amp_enabled = (os.getenv("AMP_ENABLED", "1") == "1") and (self.device.type == "cuda")

        req = (os.getenv("AMP_DTYPE", "bf16") or "bf16").lower()
        want_bf16 = req in ("bf16", "bfloat16")
        want_fp16 = req in ("fp16", "float16", "16")

        if self.amp_enabled and want_bf16 and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif self.amp_enabled:
            # fallback fp16
            self.amp_dtype = torch.float16
            if want_bf16 and (not torch.cuda.is_bf16_supported()):
                print("[Warn] bf16 requested but not supported; fallback to fp16.", flush=True)
        else:
            # AMP disabled -> dtype doesn't matter, keep fp16 placeholder
            self.amp_dtype = torch.float16

        # TF32 (Ampere)
        if os.getenv("LOCAL_TF32", "1") == "1" and self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        # preload vocab size
        temp_loader = RealDataLoader(block_size=64, batch_size=4)

        self.moe_cfg = MakeMoEConfig()
        self.moe_cfg.vocab_size = temp_loader.vocab_size
        self.moe_cfg.n_embed = EMB_DIM
        self.moe_cfg.num_experts = NUM_EXPERTS
        self.moe_cfg.top_k = TOP_K
        self.moe_cfg.block_size = int(os.getenv("BLOCK_SIZE", "64"))

        split_idx = min(1, self.moe_cfg.n_layer - 1)
        self.adapter = MakeMoEAdapter(self.moe_cfg, split_layer_idx=split_idx)

        self.pre = self.adapter.get_pre_stage().to(self.device)
        self.post = self.adapter.get_post_stage().to(self.device)
        self.experts = nn.ModuleList([self.adapter.get_expert_stage(i).to(self.device) for i in range(NUM_EXPERTS)])

        self.opt_pre = torch.optim.AdamW(self.pre.parameters(), lr=float(os.getenv("LR_PRE", "1e-3")))
        self.opt_post = torch.optim.AdamW(self.post.parameters(), lr=float(os.getenv("LR_POST", "1e-3")))
        self.opt_exps = [torch.optim.AdamW(e.parameters(), lr=float(os.getenv("LR_EXP", "1e-3"))) for e in self.experts]

        # keep CommManager for compatibility (won't be used in fast bypass)
        self.comm = CommManager()

        # fine-grained locks
        self.cache_lock = asyncio.Lock()
        self.mem_lock = asyncio.Lock()

        self.opt_pre_lock = asyncio.Lock()
        self.opt_post_lock = asyncio.Lock()
        self.opt_exp_locks = [asyncio.Lock() for _ in range(NUM_EXPERTS)]

        # in-memory KV for local bypass
        self._mem_kv: Dict[str, Dict[str, Any]] = {}
        self._loss_cache: Dict[str, Dict[str, Any]] = {}

        self.price_rate = float(os.getenv("LOCAL_PRICE_USD_PER_MS", "0.00000021"))

        print(
            f"[LocalExecutor] fast_bypass_comm={self.fast_bypass_comm}, "
            f"serialize_tensors={self.serialize_tensors}, direct_tensor_io={self.direct_tensor_io}, "
            f"amp_enabled={self.amp_enabled}, amp_dtype={self.amp_dtype}"
        )

    def _maybe_tensor(self, v: Any) -> Optional[torch.Tensor]:
        """Convert packed tensor OR torch.Tensor to tensor on self.device."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.to(self.device, non_blocking=True)
        # fallback: packed -> tensor
        return _to_tensor(v, device=self.device)

    def _unpack_tensor(self, payload: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
        if payload is None or key not in payload:
            return None
        return self._maybe_tensor(payload[key])

    async def _save_tensor(self, key: str, data: Dict[str, Any], mode: str, force_hot: bool = False):
        m = (mode or "").lower()

        # shared：永远走 CommManager 的 shared 通道（独立内存级高速链路）
        if m == "shared":
            self.comm.send_shared(key, data)
            return

        # 其它：允许 fast bypass（提升本地 GPU 利用率）
        if self.fast_bypass_comm:
            async with self.mem_lock:
                self._mem_kv[key] = data
            return

        if force_hot or m != "cold":
            self.comm.send_hot(key, data)
        else:
            self.comm.send_cold(key, data)

    async def _load_tensor(self, key: str, mode: str, delete: bool = True, try_hot_first: bool = False):
        m = (mode or "").lower()

        if m == "shared":
            return self.comm.pull_shared(key, delete=delete)

        if self.fast_bypass_comm:
            async with self.mem_lock:
                data = self._mem_kv.get(key)
                if data is None:
                    return None
                if delete:
                    del self._mem_kv[key]
                return data

        target_mode = m if m in ["hot", "cold"] else "hot"
        if try_hot_first:
            data = self.comm.pull_hot(key, delete=delete)
            if data is not None:
                return data
        if target_mode == "cold":
            return self.comm.pull_cold(key, delete=delete)
        return self.comm.pull_hot(key, delete=delete)

    def _maybe_pack(self, t: torch.Tensor) -> Any:
        """Return tensor directly in direct_tensor_io mode, else pack it."""
        if self.direct_tensor_io:
            return t
        return tensor_to_pack(t)

    async def run(self, func_name: str, path: str, payload: Dict[str, Any], mode: str = "http") -> Dict[str, Any]:
        t_start = time.perf_counter()

        trace_id = payload.get("trace_id") if isinstance(payload, dict) else None
        if not trace_id:
            trace_id = f"local_{uuid.uuid4()}"

        res_data: Dict[str, Any] = {"ok": False}

        # --------------------------
        # Forward
        # --------------------------
        if path == PATH_FWD:
            # 💡 1. 动态检测当前环境是否有 GPU
            use_cuda = torch.cuda.is_available()
            my_device_type = 'cuda' if use_cuda else 'cpu'
            if "pre" in func_name:
                x = self._unpack_tensor(payload, "x")

                # autocast forward
                with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    res = self.pre(x)

                # save for pre_bwd
                save_key = f"{trace_id}_pre"
                # store x directly in direct mode, else store packed
                await self._save_tensor(save_key,
                                        {"x": x.detach() if self.direct_tensor_io else tensor_to_pack(x.detach())},
                                        mode)

                # build output (direct tensors or packs)
                expert_inputs = {}
                for eid, t in res["expert_inputs"].items():
                    expert_inputs[eid] = t if self.direct_tensor_io else tensor_to_pack(t)

                ctx = res["context"]
                res_data = {
                    "ok": True,
                    "trace_id": trace_id,
                    "expert_inputs": expert_inputs,
                    "context": {
                        "residual": ctx["residual"] if self.direct_tensor_io else tensor_to_pack(ctx["residual"]),
                        "topk_idx": ctx["topk_idx"] if self.direct_tensor_io else tensor_to_pack(ctx["topk_idx"]),
                        "topk_weights": ctx["topk_weights"] if self.direct_tensor_io else tensor_to_pack(
                            ctx["topk_weights"]),
                    },
                    "h": res["hidden_states"] if self.direct_tensor_io else tensor_to_pack(res["hidden_states"]),
                    "topk_idx": res["expert_indices"] if self.direct_tensor_io else tensor_to_pack(
                        res["expert_indices"]),
                    "topk_vals": res["expert_weights"] if self.direct_tensor_io else tensor_to_pack(
                        res["expert_weights"]),
                }

            elif "expert" in func_name:
                eid = int(func_name.split(":")[-1])
                inp = self._unpack_tensor(payload, "inp")

                with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    out = self.experts[eid](inp)

                save_key = f"{trace_id}_exp_{eid}"
                await self._save_tensor(
                    save_key,
                    {"inp": inp.detach() if self.direct_tensor_io else tensor_to_pack(inp.detach())},
                    mode
                )

                res_data = {"ok": True, "trace_id": trace_id,
                            "out": out if self.direct_tensor_io else tensor_to_pack(out)}

            elif "post" in func_name:
                # NOTE: In split training, we must create "leaf" tensors for expert_outs and residual
                # so that post_bwd can extract grads for each expert output and residual branch.
                results = payload.get("expert_results", [])
                expert_outs: List[torch.Tensor] = []
                for r in results:
                    if isinstance(r, dict) and "out" in r:
                        t = self._maybe_tensor(r["out"])
                    else:
                        t = self._unpack_tensor(r, "out")
                    # make leaf for gradient extraction in post_bwd
                    t = t.detach().requires_grad_(True)
                    expert_outs.append(t)

                ctx_raw = payload.get("pre_context", {}) or {}
                residual = self._unpack_tensor(ctx_raw, "residual")
                topk_idx = self._unpack_tensor(ctx_raw, "topk_idx")
                topk_weights = self._unpack_tensor(ctx_raw, "topk_weights")

                # residual also needs to be leaf for grad extraction
                residual_leaf = residual.detach().requires_grad_(True) if residual is not None else None

                context = {
                    "residual": residual_leaf,
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                }
                y = self._unpack_tensor(payload, "targets")

                with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    logits, loss = self.post(expert_outs, context, targets=y)

                # ---- NaN/Inf guard: NEVER let NaN enter backward/step ----
                if (loss is None) or (not torch.isfinite(loss).all()):
                    # 不缓存 loss，不允许 post_bwd 执行
                    res_data = {
                        "ok": True,
                        "trace_id": trace_id,
                        "logits": logits.detach() if self.direct_tensor_io else tensor_to_pack(logits.detach()),
                        "acc5": float("nan") if self.acc_fill_nan else 0.0,
                        "loss": float("nan"),
                        "nan_loss": True,
                    }
                    # 直接 return（避免下面写 cache）
                    actual_ms = (time.perf_counter() - t_start) * 1000.0
                    res_data.update({
                        "actual_lat": actual_ms, "comp_lat": actual_ms, "net_lat": 0.0, "queue_lat": 0.0,
                        "cost_usd": actual_ms * self.price_rate, "is_cold": False, "cold_penalty": 0.0,
                        "func_name": func_name
                    })
                    return res_data

                async with self.cache_lock:
                    self._loss_cache[trace_id] = {
                        "loss": loss,
                        "expert_outs_leaf": expert_outs,
                        "residual_leaf": residual_leaf,
                    }

                # acc5（低频计算；没算则 NaN/0 取决于 ACC_FILL_NAN）
                acc5_val = float("nan") if getattr(self, "acc_fill_nan", False) else 0.0
                self._acc_counter += 1
                do_acc = (self.compute_acc_every > 0) and (self._acc_counter % self.compute_acc_every == 0) and (
                            y is not None)
                if do_acc:
                    try:
                        # 💡 无论 logits 和 y 是保持原样还是被提前展平了，
                        # 这里统一把它们强制展平对齐，计算整个 Batch 所有 token 的准确率
                        lt = logits.view(-1, logits.size(-1))  # 强制变为 (N, VocabSize)
                        yt = y.view(-1)  # 强制变为 (N,)

                        top5 = torch.topk(lt, k=5, dim=-1).indices  # (N, 5)
                        acc5_val = float((top5 == yt.unsqueeze(-1)).any(dim=-1).float().mean().item())
                    except Exception as e:
                        # 💡 加上打印报错信息，防止以后再有错误“死得不明不白”
                        print(f"[ACC Error] {e}", flush=True)
                        acc5_val = float("nan") if getattr(self, "acc_fill_nan", False) else 0.0

                # 返回给 controller（run_microbatch 会从这里读 loss/acc5）
                res_data = {
                    "ok": True,
                    "trace_id": trace_id,
                    "logits": logits.detach() if self.direct_tensor_io else tensor_to_pack(logits.detach()),
                    "loss": float(loss.detach().item()),
                    "acc5": acc5_val,
                    "nan_loss": False,
                }
                actual_ms = (time.perf_counter() - t_start) * 1000.0
                res_data.update({
                    "actual_lat": actual_ms, "comp_lat": actual_ms, "net_lat": 0.0, "queue_lat": 0.0,
                    "cost_usd": actual_ms * self.price_rate, "is_cold": False, "cold_penalty": 0.0,
                    "func_name": func_name
                })
                return res_data

        # --------------------------
        # Backward
        # --------------------------
        elif path == PATH_BWD:
            if "post" in func_name:
                # post backward: backprop loss and extract grads for each expert_out and residual leaf tensor
                async with self.cache_lock:
                    cache = self._loss_cache.get(trace_id)

                if cache is None:
                    res_data = {"ok": False, "error": "loss_cache missing"}
                else:
                    # guard again
                    if (cache.get("loss", None) is None) or (not torch.isfinite(cache["loss"]).all()):
                        res_data = {"ok": False, "error": "nan_loss_cached"}
                    else:
                        async with self.opt_post_lock:
                            self.opt_post.zero_grad(set_to_none=True)
                            cache["loss"].backward()

                    grads_out_by_eid: Dict[str, Any] = {}
                    eids = cache.get("expert_eids", [])
                    for i, t in enumerate(cache.get("expert_outs_leaf", [])):
                        g = t.grad
                        if g is None:
                            continue
                        key = str(eids[i]) if i < len(eids) else str(i)
                        grads_out_by_eid[key] = g if self.direct_tensor_io else tensor_to_pack(g)

                    grad_residual = None
                    rleaf = cache.get("residual_leaf", None)
                    if rleaf is not None and rleaf.grad is not None:
                        grad_residual = rleaf.grad if self.direct_tensor_io else tensor_to_pack(rleaf.grad)

                    res_data = {"ok": True, "grads_out_by_eid": grads_out_by_eid, "grad_residual": grad_residual}

                    async with self.cache_lock:
                        self._loss_cache.pop(trace_id, None)

            elif "expert" in func_name:
                try:
                    eid = int(func_name.split(":")[-1])
                    save_key = f"{trace_id}_exp_{eid}"
                    saved_data = await self._load_tensor(save_key, mode, delete=True, try_hot_first=True)

                    if not saved_data:
                        res_data = {"ok": False, "error": "expert trace not found"}
                    else:
                        inp_saved = saved_data["inp"]
                        inp = self._maybe_tensor(inp_saved).detach().requires_grad_(True)

                        with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                            out = self.experts[eid](inp)

                        grad_out = self._unpack_tensor(payload, "grad_out")
                        if grad_out is None:
                            grad_out = torch.zeros_like(out)

                        accumulate = bool(payload.get("accumulate", False))

                        async with self.opt_exp_locks[eid]:
                            if not accumulate:
                                self.opt_exps[eid].zero_grad(set_to_none=True)
                            out.backward(grad_out)

                        res_data = {
                            "ok": True,
                            "eid": eid,
                            "grad_inp": inp.grad if self.direct_tensor_io else tensor_to_pack(inp.grad),
                        }
                except Exception as e:
                    res_data = {"ok": False, "error": str(e)}
            elif "pre" in func_name:
                save_key = f"{trace_id}_pre"
                saved_data = await self._load_tensor(save_key, mode, delete=True)
                if not saved_data:
                    res_data = {"ok": False, "error": "pre trace not found"}
                else:
                    x_saved = saved_data["x"]
                    x = self._maybe_tensor(x_saved)
                    if x is not None and x.dtype != torch.long:
                        x = x.long()

                    grad_residual_pack = payload.get("grad_residual", None) if isinstance(payload, dict) else None
                    grad_inp_by_eid = payload.get("grad_inp_by_eid", {}) if isinstance(payload, dict) else {}

                    with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                        res = self.pre(x)
                        expert_inputs = res.get("expert_inputs", {}) or {}
                        ctx = res.get("context", {}) or {}
                        residual = ctx.get("residual", None)

                    outs: List[torch.Tensor] = []
                    grads: List[torch.Tensor] = []

                    # residual branch grad
                    if residual is not None and grad_residual_pack is not None:
                        outs.append(residual)
                        grads.append(self._maybe_tensor(grad_residual_pack))

                    # each expert input grad
                    for eid, inp in expert_inputs.items():
                        g_pack = grad_inp_by_eid.get(str(eid))
                        if g_pack is None:
                            continue
                        outs.append(inp)
                        grads.append(self._maybe_tensor(g_pack))

                    async with self.opt_pre_lock:
                        self.opt_pre.zero_grad(set_to_none=True)
                        if outs:
                            torch.autograd.backward(outs, grads)
                    res_data = {"ok": True}
        # --------------------------
        # Step / Zero
        # --------------------------
        elif path in [PATH_STEP, PATH_ZERO]:
            # Optimizer step / zero (no GradScaler here; bf16 recommended for stability)
            is_step = (path == PATH_STEP)

            clip = float(os.getenv("GRAD_CLIP_NORM", "1.0"))
            do_clip = (clip > 0.0) and is_step

            if "pre" in func_name:
                async with self.opt_pre_lock:
                    if is_step:
                        if do_clip:
                            torch.nn.utils.clip_grad_norm_(self.pre.parameters(), clip)
                        self.opt_pre.step()
                    else:
                        self.opt_pre.zero_grad(set_to_none=True)

            if "post" in func_name:
                async with self.opt_post_lock:
                    if is_step:
                        if do_clip:
                            torch.nn.utils.clip_grad_norm_(self.post.parameters(), clip)
                        self.opt_post.step()
                    else:
                        self.opt_post.zero_grad(set_to_none=True)

            if "expert" in func_name:
                try:
                    eid = int(func_name.split(":")[-1])
                    async with self.opt_exp_locks[eid]:
                        if is_step:
                            if do_clip:
                                torch.nn.utils.clip_grad_norm_(self.experts[eid].parameters(), clip)
                            self.opt_exps[eid].step()
                            self.opt_exps[eid].zero_grad(set_to_none=True)
                        else:
                            self.opt_exps[eid].zero_grad(set_to_none=True)
                except Exception:
                    pass

            res_data = {"ok": True}

        # --------------------------
        # Attach timing/cost fields
        # --------------------------
        actual_ms = (time.perf_counter() - t_start) * 1000.0
        if isinstance(res_data, dict):
            res_data.update(
                {
                    "actual_lat": actual_ms,
                    "comp_lat": actual_ms,
                    "net_lat": 0.0,
                    "queue_lat": 0.0,
                    "cost_usd": actual_ms * self.price_rate,
                    "is_cold": False,
                    "cold_penalty": 0.0,
                    "func_name": func_name,
                }
            )
        return res_data


# ============================================================
# Load instances / func map
# ============================================================

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


INSTANCES = _load_json(INSTANCES_FILE)
FUNC_MAP = _load_json(FUNC_MAP_FILE)
INST_BY_ID: Dict[str, Dict[str, Any]] = {x["id"]: x for x in INSTANCES}


def _inst_url(inst: Dict[str, Any]) -> str:
    # In local compute mode, URL is just a dummy ID
    if "url" in inst and inst["url"]:
        return str(inst["url"]).rstrip("/")
    return f"http://dummy"


# ============================================================
# Simulated Autoscaler (UNCHANGED)
# ============================================================
class SimulatedAutoscaler:
    def __init__(self):
        self.instance_states: Dict[str, str] = {}
        self.last_used_time: Dict[str, float] = {}
        self.func_pools: Dict[str, List[str]] = {}
        for func_name, inst_ids in FUNC_MAP.items():
            self.func_pools[func_name] = inst_ids
            for i, inst_id in enumerate(inst_ids):
                if inst_id not in self.instance_states:
                    if i < AUTOSCALE_MIN_REPLICAS:
                        self.instance_states[inst_id] = "ACTIVE"
                        self.last_used_time[inst_id] = time.time()
                    else:
                        self.instance_states[inst_id] = "OFFLINE"

    def get_active_candidates(self, func_name: str) -> List[Dict[str, Any]]:
        if not AUTOSCALE_ENABLE:
            return [INST_BY_ID[i] for i in FUNC_MAP.get(func_name, []) if i in INST_BY_ID]
        all_ids = self.func_pools.get(func_name, [])
        active_cands = []
        for iid in all_ids:
            st = self.instance_states.get(iid, "OFFLINE")
            if st in ["ACTIVE", "STARTING"]:
                if iid in INST_BY_ID:
                    active_cands.append(INST_BY_ID[iid])
        if not active_cands and all_ids:
            first_id = all_ids[0]
            self.activate_instance(first_id)
            if first_id in INST_BY_ID:
                active_cands.append(INST_BY_ID[first_id])
        return active_cands

    def activate_instance(self, inst_id: str):
        if self.instance_states.get(inst_id) == "OFFLINE":
            self.instance_states[inst_id] = "STARTING"
            self.last_used_time[inst_id] = time.time()

    def report_metric(self, func_name: str, queue_ms: float):
        if not AUTOSCALE_ENABLE: return
        if queue_ms > AUTOSCALE_QUEUE_TH_MS:
            all_ids = self.func_pools.get(func_name, [])
            for iid in all_ids:
                if self.instance_states.get(iid) == "OFFLINE":
                    self.activate_instance(iid)
                    break

    def check_starting_status(self, inst_id: str) -> bool:
        if not AUTOSCALE_ENABLE: return False
        st = self.instance_states.get(inst_id, "OFFLINE")
        if st == "STARTING":
            self.instance_states[inst_id] = "ACTIVE"
            return True
        return False

    def touch(self, inst_id: str):
        self.last_used_time[inst_id] = time.time()

    def step(self):
        if not AUTOSCALE_ENABLE: return
        now = time.time()
        for iid, st in self.instance_states.items():
            if st == "ACTIVE":
                last = self.last_used_time.get(iid, now)
                if now - last > AUTOSCALE_IDLE_TIMEOUT_S:
                    self.instance_states[iid] = "OFFLINE"


AUTOSCALER = SimulatedAutoscaler()


# ============================================================
# Helpers (Heatmap, InstanceManager, TraceCalibrator, etc.)
# ============================================================

class HotColdHeatmap:
    """
    EWMA 热度（专家访问强度）：
      heat[e] = decay * heat[e] + (1-decay) * add[e]
    然后用 top-K (HOTSET_SIZE) 作为 hot set。
    可选：进入/退出阈值 + 最短驻留，避免抖动（开关可控）。
    """

    def __init__(self, num_experts: int):
        self.num_experts = int(num_experts)
        self.decay = float(os.getenv("HEAT_EWMA_DECAY", os.getenv("HEATMAP_DECAY", "0.98")))
        self.min_prob = float(os.getenv("HEAT_MIN_PROB", os.getenv("HEATMAP_MIN_PROB", "0.01")))

        # optional anti-flap
        self.use_hysteresis = os.getenv("HEAT_HYSTERESIS", "1") == "1"
        self.hot_enter_p = float(os.getenv("HOT_ENTER_P", "0.20"))
        self.hot_exit_p = float(os.getenv("HOT_EXIT_P", "0.12"))
        self.min_stay_steps = int(os.getenv("HOT_MIN_STAY_STEPS", "30"))

        self.update_every = int(os.getenv("HEATMAP_UPDATE_EVERY", "1"))

        self.heat = np.ones(self.num_experts, dtype=np.float32) / self.num_experts
        self._step = 0
        self._is_hot = np.zeros(self.num_experts, dtype=np.int32)
        self._stay = np.zeros(self.num_experts, dtype=np.int32)

    def update_from_routing(self, topk_idx: torch.Tensor, topk_vals: torch.Tensor):
        self._step += 1
        if self.update_every > 1 and (self._step % self.update_every != 0):
            return

        idx = topk_idx.reshape(-1).detach().cpu().numpy()
        vals = topk_vals.reshape(-1).detach().cpu().numpy()

        add = np.zeros(self.num_experts, dtype=np.float32)
        for e, v in zip(idx, vals):
            ei = int(e)
            if 0 <= ei < self.num_experts:
                add[ei] += float(v)

        self.heat *= self.decay
        self.heat += (1.0 - self.decay) * add

        # avoid zeros
        self.heat = np.maximum(self.heat, self.min_prob)

        # normalize
        s = float(self.heat.sum() + 1e-9)
        self.heat /= s

        if not self.use_hysteresis:
            # hot label will be derived by top-k in hot_set()
            return

        mx = float(np.max(self.heat) + 1e-9)
        enter_th = self.hot_enter_p * mx
        exit_th = self.hot_exit_p * mx

        for i in range(self.num_experts):
            if self._is_hot[i]:
                self._stay[i] += 1
                if self._stay[i] >= self.min_stay_steps and self.heat[i] <= exit_th:
                    self._is_hot[i] = 0
                    self._stay[i] = 0
            else:
                if self.heat[i] >= enter_th:
                    self._is_hot[i] = 1
                    self._stay[i] = 0

    def hot_set(self, k: int) -> List[int]:
        k = max(1, min(int(k), self.num_experts))

        # prefer stable-hot if hysteresis enabled
        if self.use_hysteresis:
            stable = [i for i in range(self.num_experts) if self._is_hot[i] == 1]
            stable = sorted(stable, key=lambda i: float(self.heat[i]), reverse=True)
            if len(stable) >= k:
                return stable[:k]

        # fill by top heat
        order = list(range(self.num_experts))
        order.sort(key=lambda i: float(self.heat[i]), reverse=True)
        return order[:k]

    def snapshot(self) -> Dict[str, Any]:
        return {"heat": self.heat.copy(), "is_hot": self._is_hot.copy()}

HEATMAP = HotColdHeatmap(NUM_EXPERTS)
HEATMAP_LOCK = asyncio.Lock()
PREV_HOT_SET = None

def _mode_net_multiplier(mode: str) -> float:
    m = (mode or "").lower()
    if m == "hot": return HOT_NET_MUL
    if m == "cold": return COLD_NET_MUL
    if m == "shared": return SHARED_NET_MUL
    if m == "http": return HTTP_NET_MUL
    if m == "fallback": return FALLBACK_NET_MUL
    return HTTP_NET_MUL

class InstanceManager:
    def __init__(self):
        self.last_used_ts_ms: Dict[str, float] = {}
        self.lock = asyncio.Lock()
        self.keep_alive_ms = float(os.getenv("KEEP_ALIVE_MS", "300"))
        self.eviction_base_prob = float(os.getenv("EVICTION_BASE_PROB", "0.3"))
        self.eviction_tau_ms = float(os.getenv("EVICTION_TAU_MS", "2000"))
        self.keepalive_mul_hot = float(os.getenv("KEEPALIVE_MUL_HOT", "1.5"))
        self.keepalive_mul_cold = float(os.getenv("KEEPALIVE_MUL_COLD", "0.7"))
        self.keepalive_mul_http = float(os.getenv("KEEPALIVE_MUL_HTTP", "1.0"))

    def _keepalive_mul(self, mode: str) -> float:
        m = (mode or "").lower()
        if m == "hot": return self.keepalive_mul_hot
        if m == "cold": return self.keepalive_mul_cold
        return self.keepalive_mul_http

    def _default_cold_start_ms(self, func_name: str, inst: Dict[str, Any]) -> float:
        fn = (func_name or "").lower()
        region = str(inst.get("region", "local")).lower()
        is_local = ("local" in region)
        if "expert" in fn: return 80.0 if is_local else 450.0
        if "pre_" in fn or "post_" in fn or "pre" in fn or "post" in fn: return 40.0 if is_local else 250.0
        if "apply_grad" in fn or "grad" in fn: return 60.0 if is_local else 300.0
        return 40.0 if is_local else 250.0

    async def cold_start_ms(self, inst: Dict[str, Any], *, func_name: str, mode: str) -> float:
        inst_id = str(inst.get("id", ""))
        is_vm_starting = AUTOSCALER.check_starting_status(inst_id)
        AUTOSCALER.touch(inst_id)
        vm_delay = VM_COLD_START_MS if is_vm_starting else 0.0
        meta = inst.get("meta", {}) or {}
        raw = meta.get("cold_start_ms", None)
        if raw is None:
            cold_ms = self._default_cold_start_ms(func_name, inst)
        else:
            try:
                cold_ms = float(raw)
            except:
                cold_ms = self._default_cold_start_ms(func_name, inst)
        now_ms = time.perf_counter() * 1000.0
        async with self.lock:
            last_ms = self.last_used_ts_ms.get(inst_id)
            idle_ms = None if last_ms is None else max(0.0, now_ms - last_ms)
            keepalive_ms = self.keep_alive_ms * self._keepalive_mul(mode)
            is_cold = (last_ms is None) or (idle_ms is not None and idle_ms > keepalive_ms)
            if not is_cold and idle_ms is not None:
                tau = max(1.0, self.eviction_tau_ms)
                base = min(max(self.eviction_base_prob, 0.0), 1.0)
                p = base + (1.0 - base) * (1.0 - math.exp(-idle_ms / tau))
                if random.random() < p: is_cold = True
            self.last_used_ts_ms[inst_id] = now_ms
        return (cold_ms if is_cold else 0.0) + vm_delay


INSTANCE_MGR = InstanceManager()

# Trace Calibrator (Azure/Alibaba)
USE_TRACE_CALIB = os.getenv("USE_TRACE_CALIB", "0") == "1"
# USE_TRACE_CALIB = False
AZURE_PROFILE_PATH = os.getenv("AZURE_PROFILE_PATH", os.path.join("tools", "calib", "azure2021_profile.json"))
ALIBABA_GPU_PROFILE_PATH = os.getenv("ALIBABA_GPU_PROFILE_PATH",
                                     os.path.join("tools", "calib", "alibaba2025_gpu_profile.json"))
COLD_P50_MS = float(os.getenv("COLD_P50_MS", "200"))
COLD_P90_MS = float(os.getenv("COLD_P90_MS", "600"))
COLD_P99_MS = float(os.getenv("COLD_P99_MS", "3000"))
COLD_MAX_MS = float(os.getenv("COLD_MAX_MS", "15000"))
GPU_POOL_SIZE_ENV = os.getenv("GPU_POOL_SIZE", "").strip()


def _z(p: float) -> float:
    if abs(p - 0.90) < 1e-6: return 1.281551565545
    if abs(p - 0.95) < 1e-6: return 1.644853626951
    if abs(p - 0.99) < 1e-6: return 2.326347874041
    if abs(p - 0.50) < 1e-6: return 0.0
    return 0.0


def _sample_lognormal_from_q(q50: float, q90: float, rng: random.Random) -> float:
    import math
    q50 = max(float(q50), 1e-12)
    q90 = max(float(q90), q50 * 1.000001)
    mu = math.log(q50)
    sigma = (math.log(q90) - mu) / max(_z(0.90), 1e-9)
    return float(math.exp(rng.gauss(mu, sigma)))


class TraceCalibrator:
    def __init__(self):
        self.ok = False
        self.azure = None
        self.gpu = None
        self.rng = random.Random(int(os.getenv("TRACE_SEED", "123")))
        self.azure_pairs, self.azure_w = [], []
        self.gpu_pairs, self.gpu_w = [], []
        self.gpu_cap_p95 = None
        try:
            if USE_TRACE_CALIB and os.path.exists(AZURE_PROFILE_PATH):
                with open(AZURE_PROFILE_PATH, "r", encoding="utf-8") as f:
                    self.azure = json.load(f)
                pairs = self.azure.get("pairs", {}) or {}
                for k, v in pairs.items():
                    c = int(v.get("count", 0) or 0)
                    if c > 0:
                        self.azure_pairs.append(k)
                        self.azure_w.append(c)
            if USE_TRACE_CALIB and os.path.exists(ALIBABA_GPU_PROFILE_PATH):
                with open(ALIBABA_GPU_PROFILE_PATH, "r", encoding="utf-8") as f:
                    self.gpu = json.load(f)
                pairs = self.gpu.get("pairs", {}) or {}
                for k, v in pairs.items():
                    c = int(v.get("count", 0) or 0)
                    if c > 0:
                        self.gpu_pairs.append(k)
                        self.gpu_w.append(c)
                cap = (self.gpu.get("capacity_recommendation", {}) or {})
                self.gpu_cap_p95 = cap.get("cap_p95", None)
            self.ok = (self.azure is not None) or (self.gpu is not None)
        except Exception as e:
            print(f"[TraceCalib] load failed: {e}")
            self.ok = False

    def _weighted_choice(self, keys, weights):
        if not keys: return None
        return self.rng.choices(keys, weights=weights, k=1)[0]

    def sample_azure_pair(self):
        return self._weighted_choice(self.azure_pairs, self.azure_w)

    def inferred_cold_prob(self, pair_key):
        if not self.azure: return 0.0
        pairs = self.azure.get("pairs", {}) or {}
        if pair_key and pair_key in pairs:
            p = pairs[pair_key].get("inferred_cold_prob", None)
            if p is not None: return float(p)
        g = (self.azure.get("global", {}) or {})
        p = g.get("inferred_cold_prob", None)
        return float(p) if p is not None else 0.0

    def sample_cold_extra_ms(self) -> float:
        x = _sample_lognormal_from_q(COLD_P50_MS, COLD_P90_MS, self.rng)
        x = min(x, COLD_MAX_MS)
        if self.rng.random() < 0.01: x = min(max(x, COLD_P99_MS), COLD_MAX_MS)
        return float(x)

    def sample_gpu_pair(self):
        return self._weighted_choice(self.gpu_pairs, self.gpu_w)

    def sample_gpu_duration_ms(self, pair_key):
        if not self.gpu: return None
        pairs = self.gpu.get("pairs", {}) or {}
        rec = pairs.get(pair_key, None) if (pair_key and pair_key in pairs) else None
        dur = (rec.get("duration_s") if rec else (self.gpu.get("global", {}) or {}).get("duration_s", None))
        if not dur: return None
        q50 = dur.get("log_quantiles", {}).get("q50", None) or dur.get("p50", None)
        q90 = dur.get("log_quantiles", {}).get("q90", None) or dur.get("p90", None)
        if q50 is None or q90 is None: return None
        sec = _sample_lognormal_from_q(float(q50), float(q90), self.rng)
        return float(sec * 1000.0)


TRACE = TraceCalibrator()
GPU_POOL_SEM = None
if USE_TRACE_CALIB and TRACE.gpu_cap_p95 is not None:
    try:
        pool_n = int(GPU_POOL_SIZE_ENV) if GPU_POOL_SIZE_ENV else int(TRACE.gpu_cap_p95)
        pool_n = max(1, pool_n)
        GPU_POOL_SEM = asyncio.Semaphore(pool_n)
        print(f"[TraceCalib] GPU_POOL_SEM size={pool_n}")
    except:
        GPU_POOL_SEM = None

INSTANCE_SEM: Dict[str, asyncio.Semaphore] = {}
INSTANCE_MAX_CONC_DEFAULT = int(os.getenv("INSTANCE_MAX_CONC_DEFAULT", "1"))


def _get_inst_max_conc(inst: Dict[str, Any]) -> int:
    meta = inst.get("meta", {}) or {}
    mc = meta.get("max_concurrency", None)
    if mc is not None: return max(1, int(mc))
    cpu = inst.get("cpu_cores", None)
    if cpu is not None: return max(1, int(cpu))
    return INSTANCE_MAX_CONC_DEFAULT


def _get_inst_sem(inst: Dict[str, Any]) -> asyncio.Semaphore:
    inst_id = inst.get("id")
    if inst_id not in INSTANCE_SEM:
        INSTANCE_SEM[inst_id] = asyncio.Semaphore(_get_inst_max_conc(inst))
    return INSTANCE_SEM[inst_id]


class LocalTransientError(RuntimeError):
    """Transient local failure (busy/oom). Should be retried, not crash the training loop."""
    pass


async def simulate_invoke_with_breakdown(
        inst: Dict[str, Any],
        base_compute_ms: float,
        req: Dict[str, Any],
        *,
        func_name: str,
        mode: str,
        global_step: int,
) -> Tuple[float, float, float, float, float]:
    """
    Return (total_ms, queue_ms, cold_ms, net_ms, compute_ms)

    Fixes:
    - Always returns a breakdown for any instance/region (no silent None).
    - Busy/OOM is modeled as transient failure for *local non-virtual* instances only.
    - Trace-based calibration avoids double-adding cold-start (warm -> cold, cold -> max()).
    """
    region = str(inst.get("region", "local")).lower()
    is_local = ("local" in region)

    inst_id = str(inst.get("id", ""))
    is_virtual = (inst_id == "local_v_inst")

    # transient failure injection (serverless realism)
    local_oom_prob = float(os.getenv("LOCAL_OOM_PROB", "0.01"))
    local_busy_prob = float(os.getenv("LOCAL_BUSY_PROB", "0.00"))

    if is_local and (not is_virtual):
        r = random.random()
        if (local_oom_prob > 0 and r < local_oom_prob) or (local_busy_prob > 0 and r < local_busy_prob):
            raise LocalTransientError(f"Simulated Local Busy/OOM for {inst_id}")

    meta = inst.get("meta", {}) or {}
    perf = float(meta.get("performance", DEFAULT_PERFORMANCE))
    raw_compute = float(base_compute_ms) / max(perf, 1e-6)
    compute_ms = raw_compute * random.uniform(0.90, 1.10)

    # cold start from instance manager
    cold_ms = float(await INSTANCE_MGR.cold_start_ms(inst, func_name=func_name, mode=mode))

    # detect GPU-like task (used by trace calibration)
    is_gpu_task = False
    try:
        if float(meta.get("gpu_request", 0) or 0) > 0:
            is_gpu_task = True
        if float(meta.get("gpu_limit", 0) or 0) > 0:
            is_gpu_task = True
        dev = str(meta.get("device", "")).lower()
        if ("gpu" in dev) or ("cuda" in dev):
            is_gpu_task = True
    except Exception:
        pass
    if "expert" in (func_name or "").lower():
        is_gpu_task = True

    # ---- trace calibration (optional) ----
    if USE_TRACE_CALIB and TRACE.azure is not None:
        az_pair = TRACE.sample_azure_pair()
        p_cold = float(TRACE.inferred_cold_prob(az_pair))
        # Avoid double-add cold
        if cold_ms <= 1e-9:
            if random.random() < p_cold:
                cold_ms = float(TRACE.sample_cold_extra_ms())
        else:
            cold_ms = float(max(cold_ms, TRACE.sample_cold_extra_ms()))

    if USE_TRACE_CALIB and is_gpu_task and TRACE.gpu is not None:
        gpu_pair = TRACE.sample_gpu_pair()
        sampled = TRACE.sample_gpu_duration_ms(gpu_pair)
        if sampled is not None:
            sampled = float(sampled)

            # 如果 trace 实际单位是 us（常见），做一次自动纠偏（阈值可调）
            if sampled > 20000:  # 20s 太离谱，基本说明单位错了
                sampled = sampled / 1000.0

            # 再做上限裁剪，避免 5s 这种把曲线毁掉
            compute_ms = float(min(sampled, float(os.getenv("MAX_GPU_COMPUTE_MS", "200.0"))))

    # queue + gpu pool queue
    sem = _get_inst_sem(inst)
    tq0 = time.perf_counter()
    async with sem:
        queue_ms = (time.perf_counter() - tq0) * 1000.0

        if USE_TRACE_CALIB and is_gpu_task and GPU_POOL_SEM is not None:
            tg0 = time.perf_counter()
            async with GPU_POOL_SEM:
                queue_ms += (time.perf_counter() - tg0) * 1000.0

        # network latency
        net_base = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))

        # Sanity protection against misconfigured too-small RTT when DEFAULT_NET_LATENCY_MS is set
        try:
            min_ratio = float(os.getenv("MIN_NET_LATENCY_RATIO", "0.0"))
        except Exception:
            min_ratio = 0.0
        if DEFAULT_NET_LATENCY >= 10.0 and min_ratio > 0.0 and net_base < DEFAULT_NET_LATENCY * min_ratio:
            net_base = float(DEFAULT_NET_LATENCY)

        net_ms = net_base * _mode_net_multiplier(mode) * random.uniform(0.90, 1.10)
        if (mode or "").lower() == "cold":
            net_ms += float(COLD_STORAGE_MS)

        total_ms = float(queue_ms + cold_ms + net_ms + compute_ms)
        return total_ms, float(queue_ms), float(cold_ms), float(net_ms), float(compute_ms)


try:
    import aiohttp
except:
    aiohttp = None
import requests

_HTTP_SEM = asyncio.Semaphore(max(1, HTTP_CONCURRENCY))


async def invoke_http(inst: Dict[str, Any], *, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder: Not used in Local mode
    return {}


# ============================================================
# Schedulers (Baseline, Hybrid, TriScheduler)
# ============================================================

class BaselineScheduler:
    def __init__(self):
        self.rr_ptr: Dict[str, int] = {}

    def select_random(self, inst_list):
        return random.choice(inst_list)

    def select_rr(self, func_name, inst_list):
        if not inst_list: raise RuntimeError("empty inst_list")
        p = self.rr_ptr.get(func_name, 0) % len(inst_list)
        self.rr_ptr[func_name] = p + 1
        return inst_list[p]


BASELINE_SCHED = BaselineScheduler()

# ============================================================
# Online NN + Heuristic Hybrid Scheduler + NSGA-II selection (for expert backward)
# ============================================================

# ---- Online NN hyperparams ----
NN_LR = float(os.getenv("NN_LR", "3e-4"))
NN_BATCH = int(os.getenv("NN_BATCH", "64"))
NN_WARMUP = int(os.getenv("NN_WARMUP", "200"))
NN_TRAIN_EVERY = int(os.getenv("NN_TRAIN_EVERY", "10"))
NN_BUFFER = int(os.getenv("NN_BUFFER", "5000"))

USE_NSGA_FOR_EXPERT_BWD = os.getenv("USE_NSGA_FOR_EXPERT_BWD", "1") == "1"
NSGA_K = int(os.getenv("NSGA_K", "8"))


# ---- Feature engineering ----
def _fn_kind_from_name(fn: str) -> str:
    fn = (fn or "").lower()
    if "pre" in fn and "bwd" not in fn and "apply" not in fn:
        return "pre"
    if "post" in fn and "bwd" not in fn and "apply" not in fn:
        return "post"
    if "expert" in fn and "bwd" not in fn:
        return "expert"
    if "pre" in fn and "bwd" in fn:
        return "pre_bwd"
    if "post" in fn and "bwd" in fn:
        return "post_bwd"
    if "expert" in fn and "bwd" in fn:
        return "expert_bwd"
    return "other"


class OnlineLatencyNet(nn.Module):
    """Small MLP for online latency prediction (total_ms)."""

    def __init__(self, in_dim: int = 18, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class OnlinePredictor:
    """
    Online predictor for invocation latency.
    - add_sample(feat, y_ms)
    - train_step() every NN_TRAIN_EVERY steps
    """

    def __init__(self):
        self.device = torch.device("cuda" if (torch.cuda.is_available() and str(DEVICE).startswith("cuda")) else "cpu")
        self.model = OnlineLatencyNet(in_dim=18, hid=64).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=NN_LR)

        self.buf_x: List[np.ndarray] = []
        self.buf_y: List[float] = []
        self.steps = 0

        # normalization scales (rough)
        self.comp_scale = float(os.getenv("NN_COMP_SCALE", "400.0"))
        self.net_scale = float(os.getenv("NN_NET_SCALE", "120.0"))
        self.price_scale = float(os.getenv("NN_PRICE_SCALE", "0.00001"))

    def featurize(self, inst: Dict[str, Any], *, func_name: str, mode: str, base_compute_ms: float,
                  req: Dict[str, Any]) -> np.ndarray:
        meta = inst.get("meta", {}) or {}
        perf = float(meta.get("performance", DEFAULT_PERFORMANCE))
        rtt = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))
        price = float(meta.get("price", meta.get("price_usd_ms", 0.00000021)))

        # mode one-hot: hot/cold/http/fallback
        m = (mode or "").lower()
        mode_hot = 1.0 if m == "hot" else 0.0
        mode_cold = 1.0 if m == "cold" else 0.0
        mode_http = 1.0 if m == "http" else 0.0
        mode_fb = 1.0 if m == "fallback" else 0.0

        # func kind one-hot
        k = _fn_kind_from_name(func_name)
        kind_pre = 1.0 if k == "pre" else 0.0
        kind_post = 1.0 if k == "post" else 0.0
        kind_exp = 1.0 if k == "expert" else 0.0
        kind_pre_bwd = 1.0 if k == "pre_bwd" else 0.0
        kind_post_bwd = 1.0 if k == "post_bwd" else 0.0
        kind_exp_bwd = 1.0 if k == "expert_bwd" else 0.0

        # request size proxy: tokens if present else 0
        tok = 0.0
        if isinstance(req, dict):
            for kk in ["tokens", "num_tokens", "n_tokens", "seq_len"]:
                if kk in req:
                    try:
                        tok = float(req[kk])
                        break
                    except Exception:
                        pass
        tok = tok / float(max(1, SEQ_LEN * MICRO_BATCH))

        # normalized compute/net/price
        comp = float(base_compute_ms) / max(self.comp_scale, 1e-6)
        net = float(rtt) / max(self.net_scale, 1e-6)
        pr = float(price) / max(self.price_scale, 1e-9)
        inv_perf = 1.0 / max(perf, 1e-6)

        # cold start estimate
        cold_est = float(meta.get("cold_start_ms", INSTANCE_MGR._default_cold_start_ms(func_name, inst))) / 2000.0

        # region flags
        region = str(inst.get("region", "local")).lower()
        is_local = 1.0 if "local" in region else 0.0
        is_remote = 1.0 if not ("local" in region) else 0.0

        feat = np.array([
            mode_hot, mode_cold, mode_http, mode_fb,
            kind_pre, kind_post, kind_exp, kind_pre_bwd, kind_post_bwd, kind_exp_bwd,
            tok,
            comp, net, pr, inv_perf, cold_est,
            is_local, is_remote
        ], dtype=np.float32)
        return feat

    def predict(self, feat: np.ndarray) -> Optional[float]:
        if len(self.buf_y) < NN_WARMUP:
            return None
        x = torch.from_numpy(feat).to(self.device).unsqueeze(0)
        with torch.no_grad():
            yhat = float(self.model(x).item())
        return float(max(1.0, yhat))

    def add_sample(self, feat: np.ndarray, y_ms: float):
        if y_ms is None:
            return
        try:
            y = float(y_ms)
        except Exception:
            return
        if not np.isfinite(y):
            return
        self.buf_x.append(feat)
        self.buf_y.append(y)
        if len(self.buf_y) > NN_BUFFER:
            self.buf_x = self.buf_x[-NN_BUFFER:]
            self.buf_y = self.buf_y[-NN_BUFFER:]
        self.steps += 1

    def train_step(self):
        if len(self.buf_y) < max(32, NN_WARMUP):
            return
        bs = min(NN_BATCH, len(self.buf_y))
        idx = np.random.randint(0, len(self.buf_y), size=(bs,))
        xb = torch.from_numpy(np.stack([self.buf_x[i] for i in idx], axis=0)).to(self.device)
        yb = torch.from_numpy(np.array([self.buf_y[i] for i in idx], dtype=np.float32)).to(self.device)

        self.model.train()
        pred = self.model(xb)
        loss = F.smooth_l1_loss(pred, yb)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.model.eval()


ONLINE_PRED = OnlinePredictor()


# ---- NSGA-II core (fast non-dominated sorting + crowding distance) ----
def _dominates_obj(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def _fast_non_dominated_sort(objs: np.ndarray) -> List[List[int]]:
    n = objs.shape[0]
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=np.int32)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates_obj(objs[p], objs[q]):
                S[p].append(q)
            elif _dominates_obj(objs[q], objs[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        if nxt:
            fronts.append(nxt)
        else:
            break
    return fronts


def _crowding_distance(front: List[int], objs: np.ndarray) -> Dict[int, float]:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    m = objs.shape[1]
    for k in range(m):
        vals = [(i, float(objs[i, k])) for i in front]
        vals.sort(key=lambda x: x[1])
        dist[vals[0][0]] = float("inf")
        dist[vals[-1][0]] = float("inf")
        vmin = vals[0][1]
        vmax = vals[-1][1]
        if vmax - vmin < 1e-12:
            continue
        for t in range(1, len(vals) - 1):
            dist[vals[t][0]] += (vals[t + 1][1] - vals[t - 1][1]) / (vmax - vmin)
    return dist


def _nsga2_select(objs: np.ndarray, k: int) -> List[int]:
    fronts = _fast_non_dominated_sort(objs)
    chosen: List[int] = []
    for fr in fronts:
        if len(chosen) + len(fr) <= k:
            chosen.extend(fr)
        else:
            cd = _crowding_distance(fr, objs)
            fr_sorted = sorted(fr, key=lambda i: cd[i], reverse=True)
            chosen.extend(fr_sorted[: max(0, k - len(chosen))])
            break
    return chosen


class HybridScheduler:
    """
    论文版混合调度：
      - Online NN predictor：预测 latency（在线训练）
      - Heuristic rules：cost/cold/queue/deadline 惩罚
      - Expert backward：可选 NSGA-II（lat,cost,cold）做候选过滤，再用 heuristic tie-break
    """

    def __init__(self):
        self.ema_lat: Dict[str, float] = {}
        self.ema_decay = float(os.getenv("SCHED_EMA_DECAY", "0.95"))

    def update_stats(self, *args, **kwargs):
        """
        兼容两种签名：
          1) update_stats(inst, tot_ms)
          2) update_stats(func_name, logical_id, inst, req, tot_ms)
        只做 instance 级 EWMA latency。
        """
        inst = None
        tot_ms = None

        if len(args) == 2 and isinstance(args[0], dict):
            inst, tot_ms = args[0], args[1]
        elif len(args) >= 5 and isinstance(args[2], dict):
            inst, tot_ms = args[2], args[4]
        else:
            inst = kwargs.get("inst", None)
            tot_ms = kwargs.get("tot_ms", None)

        if inst is None or tot_ms is None:
            return

        inst_id = inst.get("id")
        if not inst_id:
            return

        try:
            tot_ms = float(tot_ms)
        except Exception:
            return

        v = self.ema_lat.get(inst_id)
        if v is None:
            self.ema_lat[inst_id] = tot_ms
        else:
            self.ema_lat[inst_id] = self.ema_decay * v + (1.0 - self.ema_decay) * tot_ms

    def _heuristic_components(self, inst: Dict[str, Any], *, func_name: str, mode: str, base_compute_ms: float) ->     Tuple[float, float, float, float]:
        """
        Return (lat_ms, cost_usd, cold_ms, queue_ms).

        - FULL: uses static predictor that includes cold start + network multipliers.
        - no_heuristic: intentionally *blind* to cold/net/queue so scheduling degrades
          (this makes the ablation meaningful in plots).
        """
        meta = inst.get("meta", {}) or {}
        perf = float(meta.get("performance", DEFAULT_PERFORMANCE))

        if ABL_CFG.disable_heuristic:
            # naive model: compute-only, ignores cold + network + queue
            compute_ms = float(base_compute_ms) / max(perf, 1e-6)
            lat = float(compute_ms)
            cost = _cost_usd(inst, lat)
            return float(lat), float(cost), 0.0, 0.0

        # full heuristic: (lat,cost,cold,queue) from static predictor
        lat, cost, queue_ms, cold_ms, _net = _predict_static_total_ms_and_cost(
            inst, func_name=func_name, mode=mode, base_compute_ms=base_compute_ms
        )
        return float(lat), float(cost), float(cold_ms), float(queue_ms)

    def _score(self, lat_ms: float, cost_usd: float, cold_ms: float, queue_ms: float, *, deadline_ms: float) -> float:
        late_pen = max(0.0, lat_ms - float(deadline_ms)) / max(1.0, float(deadline_ms))
        return (
                SCHED_W_LAT * float(lat_ms)
                + SCHED_W_COST * float(cost_usd) * 1000.0
                + SCHED_W_COLD * float(cold_ms)
                + SCHED_W_QUEUE * float(queue_ms)
                + 2000.0 * float(late_pen)
        )

    def select(self, func_name: str, inst_list: List[Dict[str, Any]], *, req: Dict[str, Any], global_step: int,
               mode: str, base_compute_ms: float, deadline_ms: float) -> Tuple[Dict[str, Any], float, str]:
        kind = _fn_kind_from_name(func_name)
        use_nsga = (kind == "expert_bwd") and USE_NSGA_FOR_EXPERT_BWD and (not ABL_CFG.disable_nsga)
        if not inst_list:
            raise RuntimeError("empty inst_list")

        kind = _fn_kind_from_name(func_name)
        force_nsga = (kind == "expert_bwd")

        feats = []
        objs = []
        preds_lat = []
        preds_cost = []
        preds_cold = []
        preds_queue = []

        for inst in inst_list:
            h_lat, h_cost, h_cold, h_queue = self._heuristic_components(inst, func_name=func_name, mode=mode,
                                                                        base_compute_ms=base_compute_ms)
            feat = ONLINE_PRED.featurize(inst, func_name=func_name, mode=mode, base_compute_ms=base_compute_ms, req=req)

            # -----------------------------
            # Online NN predictor (guarded)
            # -----------------------------
            nn_lat = None if ABL_CFG.disable_online_pred else ONLINE_PRED.predict(feat)

            # Trust online only when it is stable; otherwise fallback to heuristic.
            # This is critical to make FULL consistently outperform no_online.
            if nn_lat is not None:
                try:
                    r2 = float(PRED_MONITOR.get_r2())
                    mae = float(PRED_MONITOR.get_mae())
                except Exception:
                    r2, mae = 0.0, 1e18

                r2_th = float(os.getenv("NN_TRUST_R2_TH", "0.20"))
                mae_th = float(os.getenv("NN_TRUST_MAE_TH", "200.0"))  # ms
                warm = int(os.getenv("NN_TRUST_WARMUP", "60"))

                if (global_step < warm) or (r2 < r2_th) or (mae > mae_th):
                    nn_lat = None
                else:
                    # clip to avoid catastrophic mispredictions dominating selection
                    lo = float(os.getenv("NN_CLIP_LO", "0.6"))
                    hi = float(os.getenv("NN_CLIP_HI", "1.6"))
                    nn_lat = float(np.clip(float(nn_lat), lo * float(h_lat), hi * float(h_lat)))

            lat = float(nn_lat) if (nn_lat is not None) else float(h_lat)

            feats.append((inst, feat))
            preds_lat.append(lat)
            preds_cost.append(float(h_cost))  # keep your cost model stable
            preds_cold.append(float(h_cold))
            preds_queue.append(float(h_queue))

            # 3-objective: (lat, cost, cold) minimization
            objs.append([lat, float(h_cost), float(h_cold)])

        objs = np.asarray(objs, dtype=np.float32)

        # deadline filter first
        idx_ok = [i for i in range(len(inst_list)) if preds_lat[i] <= float(deadline_ms)]
        cand_idx = idx_ok if idx_ok else list(range(len(inst_list)))

        selector = "heuristic"
        if (not ABL_CFG.disable_nsga) and force_nsga and USE_NSGA_FOR_EXPERT_BWD and len(cand_idx) > 1:
            sub = objs[cand_idx]
            picked_rel = _nsga2_select(sub, k=min(NSGA_K, len(cand_idx)))
            cand_idx = [cand_idx[i] for i in picked_rel]
            selector = "nsga2"

        # tie-break with heuristic score (hybrid)
        best_i, best_s = None, 1e18
        for i in cand_idx:
            s = self._score(preds_lat[i], preds_cost[i], preds_cold[i], preds_queue[i], deadline_ms=deadline_ms)
            if s < best_s:
                best_s, best_i = s, i

        inst = inst_list[int(best_i)]
        pred_lat = float(preds_lat[int(best_i)])
        return inst, pred_lat, selector


HYBRID_SCHED = HybridScheduler()


class DeadlineEstimator:
    def __init__(self):
        self.hist: List[float] = []

    def update(self, step_ms):
        self.hist.append(float(step_ms))
        if len(self.hist) > 200: self.hist.pop(0)

    def deadline_ms(self, step) -> float:
        if step < DEADLINE_WARMUP_STEPS or len(self.hist) < 10: return float(DEADLINE_MIN_MS)
        p = np.percentile(self.hist, DEADLINE_PCTL)
        return float(max(DEADLINE_MIN_MS, p * DEADLINE_SAFETY))


DEADLINE_EST = DeadlineEstimator()


def _cost_usd(inst: Dict[str, Any], dur_ms: float) -> float:
    meta = inst.get("meta", {}) or {}
    cents_s = float(meta.get("price_cents_s", 0.0))
    return (cents_s / 100.0) * (dur_ms / 1000.0)


SCHED_W_LAT = float(os.getenv("SCHED_W_LAT", "1.0"))
SCHED_W_COST = float(os.getenv("SCHED_W_COST", "0.15"))
SCHED_W_COLD = float(os.getenv("SCHED_W_COLD", "0.25"))
SCHED_W_QUEUE = float(os.getenv("SCHED_W_QUEUE", "0.05"))
NSGA_SEED = int(os.getenv("NSGA_SEED", "42"))


def _predict_static_total_ms_and_cost(inst, *, func_name, mode, base_compute_ms):
    meta = inst.get("meta", {}) or {}
    perf = float(meta.get("performance", DEFAULT_PERFORMANCE))
    compute_ms = float(base_compute_ms) / max(perf, 1e-6)
    net_base = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))
    net_ms = net_base * _mode_net_multiplier(mode)
    if (mode or "").lower() == "cold": net_ms += float(COLD_STORAGE_MS)
    cold_ms = INSTANCE_MGR._default_cold_start_ms(func_name, inst)
    queue_ms = 0.0
    tot_ms = float(queue_ms + cold_ms + net_ms + compute_ms)
    cost = _cost_usd(inst, tot_ms)
    return tot_ms, cost, queue_ms, cold_ms, net_ms


class TriScheduler:
    def __init__(self):
        self.rng = random.Random(NSGA_SEED)

    def _online_lat(self, inst):
        if ABL_CFG.disable_online_pred: return None
        return HYBRID_SCHED.ema_lat.get(inst.get("id"))

    def _heuristic(self, inst, *, func_name, mode, base_compute_ms):
        if getattr(ABL_CFG, "use_random_sched", False):
            import random
            # Random 模式：彻底瞎分发，给出极大的随机惩罚分
            return random.random() * 1000.0, 0.0, 0.0, 0.0
            # 👆👆👆 --- [新增代码结束] --- 👆👆👆

        if getattr(ABL_CFG, "disable_heuristic", False):
            return 1e9, 0.0, 0.0, 0.0
        if ABL_CFG.disable_heuristic: return 1e9, 0.0, 0.0, 0.0
        tot_ms, cost, queue_ms, cold_ms, _ = _predict_static_total_ms_and_cost(inst, func_name=func_name, mode=mode,
                                                                               base_compute_ms=base_compute_ms)
        return float(tot_ms), float(cost), float(cold_ms), float(queue_ms)

    @staticmethod
    def _dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _pareto_front(self, pts):
        front = []
        for i, p in enumerate(pts):
            dominated = False
            for j, q in enumerate(pts):
                if j == i: continue
                if self._dominates(q, p):
                    dominated = True;
                    break
            if not dominated: front.append(i)
        return front

    def _score(self, lat_ms, cost_usd, cold_ms, queue_ms):
        return (SCHED_W_LAT * float(lat_ms) + SCHED_W_COST * float(cost_usd) * 1000.0 + SCHED_W_COLD * float(
            cold_ms) + SCHED_W_QUEUE * float(queue_ms))

    def select(self, inst_list, *, func_name, mode, base_compute_ms, deadline_ms):
        if not inst_list: raise RuntimeError("empty inst_list")
        feats = []
        for inst in inst_list:
            h_lat, h_cost, h_cold, h_queue = self._heuristic(inst, func_name=func_name, mode=mode,
                                                             base_compute_ms=base_compute_ms)
            o_lat = self._online_lat(inst)
            lat = float(o_lat) if (o_lat is not None) else float(h_lat)
            if (o_lat is None) and ABL_CFG.disable_heuristic:
                meta = inst.get("meta", {}) or {}
                lat = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY))) * 10.0
            feats.append((lat, h_cost, h_cold, h_queue))
        ok = [i for i, (lat, _, _, _) in enumerate(feats) if lat <= float(deadline_ms)]
        cand_idx = ok if ok else list(range(len(inst_list)))

        # ===== NSGA gate：只允许 expert_bwd 使用 NSGA-II =====
        kind = _fn_kind_from_name(func_name)
        use_nsga = (kind == "expert_bwd") and USE_NSGA_FOR_EXPERT_BWD and (not ABL_CFG.disable_nsga)

        # 前向 / 非 expert_bwd：只用启发式 + NN（不走 NSGA）
        if (not use_nsga) or len(cand_idx) <= 1:
            best_i, best_s = None, 1e18
            for i in cand_idx:
                s = self._score(*feats[i])
                if s < best_s:
                    best_s, best_i = s, i
            return inst_list[int(best_i)]

        # ===== expert_bwd：NSGA-II 多目标 (lat, cost, load) =====
        objs = []
        idx_map = []
        for i in cand_idx:
            inst = inst_list[i]
            lat, cost, cold_ms, queue_ms = feats[i]

            # load = inflight / max_concurrency（越大越差）
            try:
                sem = _get_inst_sem(inst)
                maxc = _get_inst_max_conc(inst)
                free = int(getattr(sem, "_value", maxc))
                inflight = max(0, maxc - free)
                load = inflight / max(1, maxc)
            except Exception:
                load = 0.0

            objs.append([float(lat), float(cost), float(load)])
            idx_map.append(i)

        objs = np.asarray(objs, dtype=np.float64)
        pick = _nsga2_select(objs, k=min(NSGA_K, len(idx_map)))
        chosen = [idx_map[j] for j in pick]

        # 用你的 score 作为最终 tie-break（保证消融/对比口径一致）
        best_i, best_s = None, 1e18
        for i in chosen:
            s = self._score(*feats[i])
            if s < best_s:
                best_s, best_i = s, i
        return inst_list[int(best_i)]

    # controller.py 约 1100 行 class TriScheduler 内插入
    def predict(self, inst, base_compute_ms, func_name="expert", mode="hot"):
        """ 为 Fig Y 提供预测值 """
        # 调用你现有的启发式或在线预测逻辑
        h_lat, _, _, _ = self._heuristic(inst, func_name=func_name, mode=mode, base_compute_ms=base_compute_ms)
        o_lat = self._online_lat(inst)
        pred = float(o_lat) if o_lat is not None else float(h_lat)
        return min(max(pred, 1.0), 5000.0)


TRI_SCHED = TriScheduler()


# ============================================================
# Invocation Wrapper
# ============================================================
def _maybe_autoscale(func_name, candidates, queue_ms, global_step):
    AUTOSCALER.report_metric(func_name, queue_ms)


def _get_candidates(func_name: str) -> List[Dict[str, Any]]:
    return AUTOSCALER.get_active_candidates(func_name)


def _payload(obj: Any) -> dict:
    if isinstance(obj, dict) and "payload" in obj and isinstance(obj["payload"], dict): return obj["payload"]
    return obj if isinstance(obj, dict) else {}


def _to_tensor(v: Any, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        t = v
    elif isinstance(v, dict):
        try:
            t = pack_to_tensor(v)
        except:
            t = torch.as_tensor(v)
    else:
        t = torch.as_tensor(v)
    if dtype is not None: t = t.to(dtype=dtype)
    return t.to(device)


async def invoke_with_retry(
        func_name,
        logical_id,
        candidates,
        req,
        base_compute_ms,
        *,
        mode,
        max_tries,
        forced_inst=None,
        global_step=0,
):
    """
    Serverless-style invocation with retry + backoff + graceful degradation.
    Returns: (inst, meta_dict, retry_cnt)
      meta_dict contains:
        ok, func_name, actual_lat, pred_lat, cost_usd, is_cold, cold_penalty,
        net_lat, comp_lat, queue_lat, raw_breakdown, fail_reason(optional)
    """
    if not candidates:
        # In local-compute mode, allow a virtual instance so metrics still record.
        if LOCAL_EXECUTOR is not None:
            candidates = [{"id": "local_v_inst", "region": "local", "meta": {"performance": 1.0}}]
        else:
            raise RuntimeError(f"invoke candidates empty for {func_name}")

    # baseline forced instance selection (kept)
    if forced_inst is None and ABL_CFG.is_baseline:
        if ABL_CFG.use_random_sched:
            forced_inst = BASELINE_SCHED.select_random(candidates)
        elif ABL_CFG.use_rr_sched:
            forced_inst = BASELINE_SCHED.select_rr(func_name, candidates)

    # retry knobs
    backoff_ms = float(os.getenv("RETRY_BACKOFF_MS", "5.0"))
    max_backoff_ms = float(os.getenv("RETRY_BACKOFF_MAX_MS", "50.0"))
    fail_penalty_ms = float(os.getenv("FAIL_PENALTY_MS", "200.0"))

    tries = 0
    retry_cnt = 0
    last_err = None

    async def _simulate_one(inst, pred_ms_val: float, selector_name: str):
        # simulate breakdown; may raise LocalTransientError/RuntimeError
        breakdown_tuple = await simulate_invoke_with_breakdown(
            inst,
            base_compute_ms,
            req,
            func_name=func_name,
            mode=mode,
            global_step=global_step,
        )
        actual_ms, queue_lat, cold_lat, net_lat, comp_lat = breakdown_tuple
        total_ms = float(min(max(actual_ms, 0.001), 15000.0))
        # ---- online NN: add sample + periodic train ----
        try:
            feat = ONLINE_PRED.featurize(inst, func_name=func_name, mode=mode, base_compute_ms=base_compute_ms, req=req)
            ONLINE_PRED.add_sample(feat, total_ms)
            ONLINE_PRED.steps += 1
            if (not ABL_CFG.disable_online_pred) and (ONLINE_PRED.steps % NN_TRAIN_EVERY == 0):
                ONLINE_PRED.train_step()
        except Exception:
            pass

        price_rate = float(inst.get("meta", {}).get("price", 0.00000021))
        cost_usd = total_ms * price_rate

        # IMPORTANT: call scheduler update if signature matches (be defensive)
        try:
            # common signature: update_stats(func_name, logical_id, inst, req, total_ms)
            HYBRID_SCHED.update_stats(func_name, logical_id, inst, req, total_ms)
        except Exception:
            try:
                # fallback signature: update_stats(inst, total_ms)
                HYBRID_SCHED.update_stats(inst, total_ms)
            except Exception:
                pass

        try:
            _maybe_autoscale(func_name, candidates, queue_lat, global_step)
        except Exception:
            pass

        if SIM_SLEEP:
            await asyncio.sleep(total_ms / 1000.0)

        pred_cold_ms = float(INSTANCE_MGR._default_cold_start_ms(func_name, inst))
        meta = {
            "ok": True,
            "func_name": func_name,
            "actual_lat": total_ms,
            "pred_lat": float(pred_ms_val),
            "selector": str(selector_name),
            "cost_usd": float(cost_usd),
            "is_cold": float(cold_lat) > 0.0,
            "cold_penalty": float(cold_lat),
            "net_lat": float(net_lat),
            "comp_lat": float(comp_lat),
            "queue_lat": float(queue_lat),
            "raw_breakdown": breakdown_tuple,
            "pred_cold_ms": float(pred_cold_ms),
        }
        return meta

    # choose candidate list
    cand = list(candidates)

    # forced instance path
    if forced_inst is not None:
        try:
            meta = await _simulate_one(forced_inst, 0.0, "forced")
            return forced_inst, meta, retry_cnt
        except Exception as e:
            last_err = e
            # fall through to normal retry path

    while tries < max_tries and cand:
        tries += 1

        # ---- select instance (ONLINE NN + heuristic; NSGA-II for expert backward) ----
        inst = None
        pred_ms = 0.0
        selector = "heuristic"
        try:
            if not ABL_CFG.is_baseline:
                dl = float(DEADLINE_EST.deadline_ms(global_step))
                inst, pred_ms, selector = HYBRID_SCHED.select(
                    func_name, cand,
                    req=req, global_step=global_step,
                    mode=mode, base_compute_ms=base_compute_ms, deadline_ms=dl
                )
            else:
                inst = random.choice(cand)
                pred_ms = 0.0
                selector = "baseline"
        except Exception:
            inst = random.choice(cand)
            pred_ms = 0.0
            selector = "fallback"

        try:
            meta = await _simulate_one(inst, pred_ms, selector)

            return inst, meta, retry_cnt

        except LocalTransientError as e:
            last_err = e
            retry_cnt += 1

            # backoff + jitter (serverless-like)
            jitter = random.uniform(0.0, 1.0)
            sleep = min(max_backoff_ms, backoff_ms * (1.5 ** max(retry_cnt - 1, 0))) * (0.5 + jitter)
            if SIM_SLEEP:
                await asyncio.sleep(sleep / 1000.0)

            # avoid repeatedly picking the same inst
            bad = inst.get("id")
            cand = [x for x in cand if x.get("id") != bad] + [inst]
            continue

        except RuntimeError as e:
            # treat "Simulated Local Busy/OOM" as transient too (compat)
            if "Simulated Local Busy/OOM" in str(e):
                last_err = e
                retry_cnt += 1
                jitter = random.uniform(0.0, 1.0)
                sleep = min(max_backoff_ms, backoff_ms * (1.5 ** max(retry_cnt - 1, 0))) * (0.5 + jitter)
                if SIM_SLEEP:
                    await asyncio.sleep(sleep / 1000.0)
                bad = inst.get("id")
                cand = [x for x in cand if x.get("id") != bad] + [inst]
                continue
            last_err = e
            retry_cnt += 1
            # remove this inst and retry others
            bad = inst.get("id")
            cand = [x for x in cand if x.get("id") != bad]
            continue

        except Exception as e:
            last_err = e
            retry_cnt += 1
            bad = inst.get("id") if inst else None
            if bad:
                cand = [x for x in cand if x.get("id") != bad]
            continue

    # ---- Graceful degrade: do NOT crash training ----
    # approximate breakdown: compute + default net + penalty
    fallback_inst = candidates[0]
    net_lat = float(os.getenv("DEFAULT_NET_LATENCY_MS", str(DEFAULT_NET_LATENCY)))
    comp_lat = float(base_compute_ms)
    queue_lat = 0.0
    cold_lat = 0.0
    total_ms = float(queue_lat + cold_lat + net_lat + comp_lat + fail_penalty_ms)

    price_rate = float(fallback_inst.get("meta", {}).get("price", 0.00000021))
    cost_usd = total_ms * price_rate

    meta = {
        "ok": False,
        "func_name": func_name,
        "actual_lat": total_ms,
        "pred_lat": 0.0,
        "cost_usd": float(cost_usd),
        "is_cold": False,
        "cold_penalty": 0.0,
        "net_lat": net_lat,
        "comp_lat": comp_lat,
        "queue_lat": queue_lat,
        "raw_breakdown": (total_ms, queue_lat, cold_lat, net_lat, comp_lat),
        "fail_reason": str(last_err),
    }
    return fallback_inst, meta, retry_cnt


async def _invoke_fn(func_name, *, mode, base_compute_ms, http_path, payload, global_step):
    cands = _get_candidates(func_name)

    # If local executor is enabled but JSON has no mapping for this func,
    # use a virtual instance AND still simulate breakdown for realistic metrics.
    if (not cands) and LOCAL_EXECUTOR:
        inst = {"id": "local_v_inst", "region": "local", "meta": {"performance": 1.0}}
        try:
            breakdown_tuple = await simulate_invoke_with_breakdown(
                inst,
                base_compute_ms,
                req={},
                func_name=func_name,
                mode=mode,
                global_step=global_step,
            )
            total_ms, queue_lat, cold_lat, net_lat, comp_lat = breakdown_tuple
            price_rate = float(inst.get("meta", {}).get("price", 0.00000021))
            cost_usd = float(total_ms) * price_rate
            meta = {
                "ok": True,
                "func_name": func_name,
                "actual_lat": float(total_ms),
                "pred_lat": 0.0,
                "cost_usd": float(cost_usd),
                "is_cold": float(cold_lat) > 0.0,
                "cold_penalty": float(cold_lat),
                "net_lat": float(net_lat),
                "comp_lat": float(comp_lat),
                "queue_lat": float(queue_lat),
                "raw_breakdown": breakdown_tuple,
            }
        except Exception as e:
            # last resort: still don't crash
            net_lat = float(os.getenv("DEFAULT_NET_LATENCY_MS", str(DEFAULT_NET_LATENCY)))
            comp_lat = float(base_compute_ms)
            total_ms = float(net_lat + comp_lat)
            meta = {
                "ok": False,
                "func_name": func_name,
                "actual_lat": total_ms,
                "pred_lat": 0.0,
                "cost_usd": 0.0,
                "is_cold": False,
                "cold_penalty": 0.0,
                "net_lat": net_lat,
                "comp_lat": comp_lat,
                "queue_lat": 0.0,
                "raw_breakdown": (total_ms, 0.0, 0.0, net_lat, comp_lat),
                "fail_reason": str(e),
            }

        obj = await LOCAL_EXECUTOR.run(func_name, http_path, payload)
        return inst, obj, meta, 0

    if not cands:
        cands = [INST_BY_ID[i] for i in FUNC_MAP.get(func_name, []) if i in INST_BY_ID]
        if not cands:
            raise RuntimeError(f"No candidates for {func_name}")

    inst, meta, retry_cnt = await invoke_with_retry(
        func_name, 0, cands, req={}, base_compute_ms=base_compute_ms,
        mode=mode, max_tries=INVOKE_RETRIES, global_step=global_step
    )

    if LOCAL_EXECUTOR:
        obj = await LOCAL_EXECUTOR.run(func_name, http_path, payload)
    else:
        obj = await invoke_http(inst, path=http_path, payload=payload) if USE_HTTP_EXEC else {}
    return inst, obj, meta, retry_cnt


# ============================================================
# Main Train Loop (UNCHANGED logic, just shortened for brevity)
# ============================================================
# (Keeping the exact logic of run_microbatch, train, etc. from your original file,
#  but ensuring they use the new _invoke_fn which calls LOCAL_EXECUTOR)

def simulate_traffic_skew(topk_idx, topk_vals, num_experts, global_step):
    # (Same traffic skew logic as original)
    if num_experts <= 1: return topk_idx, topk_vals
    is_2d = (topk_idx.ndim == 2)
    if is_2d:
        topk_idx_3d, topk_vals_3d = topk_idx.unsqueeze(1), topk_vals.unsqueeze(1)
    else:
        topk_idx_3d, topk_vals_3d = topk_idx, topk_vals
    B, T, K = topk_idx_3d.shape
    device = topk_idx_3d.device
    phase = max(0, int(global_step // max(1, HOTSPOT_DRIFT_EVERY)))
    hot0 = (phase * max(1, HOTSPOT_SPAN)) % num_experts
    hot_set = [(hot0 + i) % num_experts for i in range(max(1, HOTSPOT_SPAN))]
    warm_e = (hot0 + max(1, HOTSPOT_SPAN)) % num_experts
    new_idx, new_vals = topk_idx_3d.clone(), topk_vals_3d.clone()
    rand_vals = torch.rand((B, T), device=device)
    mask_hot = (rand_vals < HOT_PROB)
    mask_warm = (rand_vals >= HOT_PROB) & (rand_vals < (HOT_PROB + WARM_PROB))
    mask_others = ~(mask_hot | mask_warm)
    if K >= 1:
        t0 = new_idx[..., 0]
        if mask_hot.any(): t0[mask_hot] = random.choice(hot_set)
        if mask_warm.any(): t0[mask_warm] = warm_e
        if mask_others.any(): t0[mask_others] = hot_set[0]
        new_idx[..., 0] = t0
        v0 = new_vals[..., 0]
        v0[mask_hot] = 1.0;
        v0[mask_warm] = 0.5;
        v0[mask_others] = 0.8
        new_vals[..., 0] = v0
    if K >= 2:
        t1 = new_idx[..., 1]
        target_e = hot_set[1] if len(hot_set) >= 2 else hot_set[0]
        if mask_hot.any(): t1[mask_hot] = target_e
        if mask_warm.any(): t1[mask_warm] = hot_set[0]
        if mask_others.any(): t1[mask_others] = warm_e
        new_idx[..., 1] = t1
        v1 = new_vals[..., 1]
        v1[mask_hot] = 0.9;
        v1[mask_warm] = 0.3;
        v1[mask_others] = 0.4
        new_vals[..., 1] = v1
    if is_2d: return new_idx.squeeze(1), new_vals.squeeze(1)
    return new_idx, new_vals


# Metrics / Data Loading / Train Function (Standard)
def _ensure_dir(path: str): os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


METRICS_FILE = os.getenv("METRICS_FILE", "metrics.csv")


def write_metrics_header(path: str):
    _ensure_dir(path)
    cols = ["step", "split", "loss", "acc_top1", "acc_top5", "step_time_ms", "pre_lat_ms", "post_lat_ms", "exp_lat_ms",
            "inv_total_ms", "inv_queue_ms", "inv_cold_ms", "inv_net_ms", "inv_compute_ms", "inv_retry_cnt", "hot_ratio",
            "hot_set_changed", "hot_set_jaccard", "fwd_mode_hot_frac", "fwd_mode_cold_frac", "fwd_mode_http_frac",
            "grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac", "deadline_ms",
            "deadline_violation_frac", "cost_usd_step", "cost_usd_pre_fwd", "cost_usd_post_fwd", "cost_usd_expert_fwd"]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def append_metrics(path: str, row: Dict[str, Any]): pd.DataFrame([row]).to_csv(path, mode="a", header=False,
                                                                               index=False)


def _build_vocab(txt_path, vocab_path):
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text));
        stoi = {ch: i for i, ch in enumerate(chars)}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(stoi, f, indent=2)
    return stoi, {int(i): ch for ch, i in stoi.items()}


def _load_ids(txt_path, stoi):
    with open(txt_path, "r", encoding="utf-8") as f: text = f.read()
    return torch.tensor([stoi.get(ch, 0) for ch in text], dtype=torch.long)


class TextBatcher:
    def __init__(self, ids, batch_size, seq_len, seed=42):
        self.ids = ids.contiguous();
        self.bs = int(batch_size);
        self.T = int(seq_len);
        self.rng = random.Random(seed)

    def next_batch(self):
        max_start = int(self.ids.numel() - (self.T + 1))
        starts = [self.rng.randint(0, max_start) for _ in range(self.bs)]
        x = torch.stack([self.ids[s:s + self.T] for s in starts], dim=0)
        y = torch.stack([self.ids[s + 1:s + 1 + self.T] for s in starts], dim=0)
        return x, y


def _acc_topk(logits, y, k):
    topk = torch.topk(logits, k=k, dim=-1).indices
    return float((topk == y.unsqueeze(-1)).any(dim=-1).float().mean().item())


class ExpertUpdatePolicy:
    def __init__(self, n, cold_acc):
        self.n = int(n);
        self.cold_acc = max(1, int(cold_acc));
        self.pending = [0] * self.n

    def decide(self, hot_set):
        if ABL_CFG.force_sync_update or FORCE_SYNC_UPDATE:
            self.pending = [0] * self.n;
            return list(range(self.n)), 0, 0.0
        upd, cold_upd = [], 0
        for eid in range(self.n):
            if eid in hot_set:
                upd.append(eid);
                self.pending[eid] = 0
            else:
                self.pending[eid] += 1
                if self.pending[eid] >= self.cold_acc: upd.append(eid); cold_upd += 1; self.pending[eid] = 0
        return upd, cold_upd, float(np.mean(self.pending))


POLICY = ExpertUpdatePolicy(NUM_EXPERTS, COLD_ACC_STEPS)


async def run_microbatch(step: int, micro_step: int, x: torch.Tensor, y: torch.Tensor):
    split = "train"
    trace_id = f"step_{step}_mb_{micro_step}_{uuid.uuid4().hex[:8]}"
    my_device = DEVICE

    x_tok = x.to(my_device)
    y_tok = y.to(my_device)

    all_mb_rows = []

    # 1) zero grad (pre/post/expert 的 zero 由各自执行；这里至少 pre_zero 保留你原逻辑)
    if split == "train":
        res_zero = await _invoke_fn("moe.pre_zero", mode="shared", payload={}, base_compute_ms=0,
                                    http_path=PATH_ZERO, global_step=step)
        all_mb_rows.append(res_zero[2] if isinstance(res_zero, tuple) else res_zero)

        # 建议也加 post_zero（否则 post 的梯度可能累积）
        res_zero2 = await _invoke_fn("moe.post_zero", mode="shared", payload={}, base_compute_ms=0,
                                     http_path=PATH_ZERO, global_step=step)
        all_mb_rows.append(res_zero2[2] if isinstance(res_zero2, tuple) else res_zero2)

    # ============================================================
    # 2) Forward
    # ============================================================
    # Pre fwd（必须带 trace_id）
    _, pre_obj, bd1, _ = await _invoke_fn(
        "moe.pre_fwd", mode="http",
        payload={"trace_id": trace_id, "x": tensor_to_pack(x_tok)},
        base_compute_ms=10.0, http_path=PATH_FWD, global_step=step
    )
    all_mb_rows.append(bd1)

    # ---- update heatmap & decide hot/cold ----
    try:
        async with HEATMAP_LOCK:
            ctx = pre_obj.get("context", {}) if isinstance(pre_obj.get("context", {}), dict) else {}
            topk_idx = ctx.get("topk_idx", None)
            topk_vals = ctx.get("topk_weights", None)
            if topk_idx is not None and topk_vals is not None:
                HEATMAP.update_from_routing(_to_tensor(topk_idx, device="cpu"), _to_tensor(topk_vals, device="cpu"))
            hot_set = set(HEATMAP.hot_set(HOTSET_SIZE))
    except Exception:
        hot_set = set(range(NUM_EXPERTS))
    # 💡 修复：如果是在跑 no_hotcold 消融实验，强行清空热专家池，并关闭异步更新
        # === 🚀 新增：彻底支持所有 Baseline 和 Ablation 的同步逻辑 ===
        _b_mode = os.getenv("BASELINE_MODE", "")

        # 1. 消融实验：关闭冷热感知
        if getattr(ABL_CFG, "disable_hotcold", False):
            hot_set = set()
            ABL_CFG.force_sync_update = True

        # 2. BSP (Bulk Synchronous Parallel): 强制同步屏障
        # 表现：不区分冷热，主线程必须死等所有专家（无论多慢）算完并更新参数才进入下一步。
        # 预期结果：Loss 极其平稳，但 Step Time 会被冷启动拖垮，延迟极高。
        elif _b_mode == "bsp":
            hot_set = set()
            ABL_CFG.force_sync_update = True

        # 3. ASP (Asynchronous Parallel): 纯异步放飞自我
        # 表现：不区分冷热，主线程不设等待屏障，专家算完立刻在后台异步更新自己的参数。
        # 预期结果：Step Time 极快，但因为“陈旧梯度 (Stale Gradients)”满天飞，Loss 极易发散崩盘。
        elif _b_mode == "asp":
            hot_set = set()
            ABL_CFG.force_sync_update = False
            os.environ["COLD_ACC_STEPS"] = "1"  # 强制冷专家憋气步数为1，算完立刻异步更新，绝不等待
        # =========================================================

    # decide which experts will APPLY (step) this iteration (hot 每步更新，cold 走累积/延迟)
    upd_eids, cold_upd, pending_mean = POLICY.decide(list(hot_set))

    expert_inputs = pre_obj.get("expert_inputs", {})  # {eid: pack_tensor}
    expert_eids = list(expert_inputs.keys())

    # Expert fwd（必须带 trace_id）
    fwd_tasks = []
    for eid, inp_pack in expert_inputs.items():
        fwd_tasks.append(_limited_expert(
            _invoke_fn(
                f"moe.expert_fwd:{eid}", mode=("hot" if eid in hot_set else "cold"),
                payload={"trace_id": trace_id, "inp": inp_pack},
                base_compute_ms=20.0, http_path=PATH_FWD, global_step=step
            )
        ))
    expert_results = await asyncio.gather(*fwd_tasks)

    for res in expert_results:
        all_mb_rows.append(res[2])

    exp_outs = [res[1] for res in expert_results]  # list of obj dict (contain "out")

    # Post fwd（必须带 trace_id + expert_eids）
    _, post_obj, bd3, _ = await _invoke_fn(
        "moe.post_fwd", mode="http",
        payload={
            "trace_id": trace_id,
            "expert_eids": expert_eids,  # 关键：让 post_bwd 能对齐梯度
            "expert_results": exp_outs,
            "pre_context": pre_obj.get("context", {}),
            "targets": tensor_to_pack(y_tok),
        },
        base_compute_ms=10.0, http_path=PATH_FWD, global_step=step
    )
    all_mb_rows.append(bd3)
    # -------------------------------
    # NaN/Inf guard (microbatch-level)
    # 一旦 post_fwd 得到 NaN，就直接跳过 backward + step，避免污染参数
    # -------------------------------
    loss_tmp = post_obj.get("loss", None)

    def _is_finite_number(v):
        try:
            return (v is not None) and np.isfinite(float(v))
        except Exception:
            return False

    if (post_obj.get("nan_loss", False)) or (not _is_finite_number(loss_tmp)):
        # 保险：把本 microbatch 标记为 nan_loss，并且不要进入 backward/step
        loss_val = float("nan")
        acc5_val = float("nan")

        # 这里可选：清一下 pre/post 的梯度缓存（你本 microbatch 开头已经 zero 过，但再做一次更稳）
        if split == "train":
            res_zero = await _invoke_fn("moe.pre_zero", mode="shared", payload={}, base_compute_ms=0,
                                        http_path=PATH_ZERO, global_step=step)
            all_mb_rows.append(res_zero[2] if isinstance(res_zero, tuple) else res_zero)

            res_zero2 = await _invoke_fn("moe.post_zero", mode="shared", payload={}, base_compute_ms=0,
                                         http_path=PATH_ZERO, global_step=step)
            all_mb_rows.append(res_zero2[2] if isinstance(res_zero2, tuple) else res_zero2)

        # 直接 early return：不反传、不 step
        final_cold = sum([1 for r in all_mb_rows if isinstance(r, dict) and r.get("is_cold")])
        final_total = len(all_mb_rows)
        return {
            "loss": loss_val,
            "acc5": acc5_val,
            "nan_loss": True,
            "cold_upd": 0,
            "pending_mean": float(POLICY.pending_mean() if hasattr(POLICY, "pending_mean") else 0.0),
            "rows": all_mb_rows,
            "hot_ratio": (final_total - final_cold) / final_total if final_total > 0 else 1.0,
        }

    # 正常路径才会走到这里
    loss_val = post_obj.get("loss", 0.0)
    acc5_val = post_obj.get("acc5", 0.0)

    # ============================================================
    # 3) Backward & Step
    # ============================================================
    if split == "train":
        # (A) Post backward：返回每个 expert_out 的 grad + residual 的 grad
        _, post_bwd_obj, bbd1, _ = await _invoke_fn(
            "moe.post_bwd", mode="shared",
            payload={"trace_id": trace_id},
            base_compute_ms=15.0, http_path=PATH_BWD, global_step=step
        )
        all_mb_rows.append(bbd1)

        grads_out_by_eid = post_bwd_obj.get("grads_out_by_eid", {})  # {str(eid): pack_grad}
        grad_residual = post_bwd_obj.get("grad_residual", None)  # pack_grad or None

        # (B) Expert backward：每个 expert 用自己的 grad_out
        bwd_tasks = []
        for eid in expert_eids:
            grad_out = grads_out_by_eid.get(str(eid))
            if grad_out is None:
                continue
            bwd_tasks.append(_limited_expert(
                _invoke_fn(
                    f"moe.expert_bwd:{eid}", mode=("hot" if eid in hot_set else "cold"),
                    payload={"trace_id": trace_id, "grad_out": grad_out, "accumulate": (eid not in upd_eids)},
                    base_compute_ms=30.0, http_path=PATH_BWD, global_step=step
                )
            ))
        expert_bwd_results = await asyncio.gather(*bwd_tasks)

        for res in expert_bwd_results:
            all_mb_rows.append(res[2])

        # 收集每个 expert_input 的梯度（grad_inp）
        grad_inp_by_eid = {}
        for res in expert_bwd_results:
            obj = res[1]  # expert_bwd obj
            # obj 应该带 {"eid":..., "grad_inp":...}
            if isinstance(obj, dict):
                eid = obj.get("eid", None)
                g = obj.get("grad_inp", None)
                if eid is not None and g is not None:
                    grad_inp_by_eid[str(eid)] = g

        # (C) Pre backward：用 grad_residual + grad_inp_by_eid 回传
        _, _, bbd3, _ = await _invoke_fn(
            "moe.pre_bwd", mode="shared",
            payload={
                "trace_id": trace_id,
                "grad_residual": grad_residual,
                "grad_inp_by_eid": grad_inp_by_eid,
            },
            base_compute_ms=15.0, http_path=PATH_BWD, global_step=step
        )
        all_mb_rows.append(bbd3)

        # (D) Step：一定要包含 post_step
        hot_set = HEATMAP.hot_set(HOTSET_SIZE)
        upd_eids, _, _ = POLICY.decide(hot_set)

        step_tasks = [
            _invoke_fn("moe.pre_step", mode="shared", payload={}, base_compute_ms=5, http_path=PATH_STEP,
                       global_step=step),
            _invoke_fn("moe.post_step", mode="shared", payload={}, base_compute_ms=5, http_path=PATH_STEP,
                       global_step=step),
        ]
        for eid in upd_eids:
            step_tasks.append(_limited_expert(
                _invoke_fn(f"moe.expert_step:{eid}", mode=("hot" if eid in hot_set else "cold"), payload={},
                           base_compute_ms=10,
                           http_path=PATH_STEP, global_step=step)
            ))
        s_results = await asyncio.gather(*step_tasks)
        for res in s_results:
            all_mb_rows.append(res[2])

    # ============================================================
    # 4) 汇总
    # ============================================================
    final_cold = sum([1 for r in all_mb_rows if isinstance(r, dict) and r.get("is_cold")])
    final_total = len(all_mb_rows)

    return {
        "loss": loss_val,
        "acc5": acc5_val,
        "cold_upd": cold_upd,
        "pending_mean": pending_mean,
        "rows": all_mb_rows,
        "hot_ratio": (final_total - final_cold) / final_total if final_total > 0 else 1.0,
    }


async def train():
    global logger
    # 确保初始化时清空之前的记录或保持追加逻辑
    if not os.path.exists(METRICS_FILE): write_metrics_header(METRICS_FILE)

    stoi, _ = _build_vocab(DATA_PATH, VOCAB_PATH)
    ids = _load_ids(DATA_PATH, stoi)
    total_tokens = ids.numel()
    n = int(ids.numel())
    n_train = max(SEQ_LEN + 2, int(n * 0.9))
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    train_batcher = TextBatcher(train_ids, BATCH_SIZE, SEQ_LEN, seed=SEED)
    val_batcher = TextBatcher(val_ids, BATCH_SIZE, SEQ_LEN, seed=SEED + 999)
    dataset = RealDataLoader(block_size=64, batch_size=4)

    # 计算一个 Epoch 需要多少步
    steps_per_epoch = max(1, total_tokens // (BATCH_SIZE * SEQ_LEN))

    for step in range(1, MAX_STEPS + 1):
        # 动态计算当前的 epoch
        current_epoch = (step // steps_per_epoch) + 1

        t_step0 = time.perf_counter()
        AUTOSCALER.step()
        split = "val" if (step % VAL_INTERVAL == 0) else "train"

        # 准备微批次数据
        mb = MICRO_BATCH
        xs, ys = [], []
        for _ in range(mb):
            x, y = dataset.get_batch('train')
            xs.append(x)
            ys.append(y)

        if len(xs) != mb:
            print(f">>> [Fatal Error] Loop mismatch! Generated {len(xs)}, expected {mb}")
            continue

        # 并发执行微批次任务
        # 并发执行微批次任务（限流，防止 CPU OOM）
        tasks = [_limited_mb(run_microbatch(step, i, xs[i], ys[i])) for i in range(mb)]
        results = await asyncio.gather(*tasks)

        # 展平所有调用记录
        all_invoke_results = []
        for r in results:
            if isinstance(r, dict) and "rows" in r:
                all_invoke_results.extend(r["rows"])
            elif isinstance(r, list):
                all_invoke_results.extend(r)

        # ============================================================
        # 【修正】指标计算与记录逻辑 (HeatMoE 专用 - 鲁棒版)
        # ============================================================

        # 定义辅助函数确保字典格式
        def ensure_dict(r):
            if isinstance(r, dict): return r
            if isinstance(r, (tuple, list)) and len(r) >= 3:
                res_meta = r[2]
                if isinstance(res_meta, (tuple, list)):
                    return {"actual_lat": float(res_meta[0]), "cost_usd": 0.0, "func_name": ""}
                return res_meta if isinstance(res_meta, dict) else {}
            return {}

        # 1. 预处理：清洗数据
        clean_invokes = [ensure_dict(r) for r in all_invoke_results]
        valid_mb_results = [r for r in results if r is not None]

        # 2. 更新预测监控器 (合并去重，只更新一次)
        for r in clean_invokes:
            if "actual_lat" in r and "pred_lat" in r:
                # 过滤异常值和非 FWD 调用
                func_name = r.get("func_name", "")
                is_valid_time = 1.0 < r["actual_lat"] < 5000.0  # 物理限幅
                if "fwd" in func_name and is_valid_time:
                    pred_target = (os.getenv("PRED_TARGET", "total") or "total").lower()

                    pred_lat = float(r.get("pred_lat", 0.0))
                    actual_lat = float(r.get("actual_lat", 0.0))

                    if pred_target == "nocold":
                        # 去掉真实 cold_penalty；预测侧用 pred_cold_ms 做近似剥离
                        actual_lat = max(0.0, actual_lat - float(r.get("cold_penalty", 0.0)))
                        pred_lat = max(0.0, pred_lat - float(r.get("pred_cold_ms", 0.0)))

                    PRED_MONITOR.update(pred_lat=pred_lat, actual_lat=actual_lat)

        # 获取最新的 R2 和 MAE
        current_r2, current_mae = PRED_MONITOR.get_r2_mae()

        # 3. 基础指标聚合
        loss = float(np.mean([r.get("loss", 0.0) for r in valid_mb_results])) if valid_mb_results else 0.0
        if valid_mb_results:
            acc_list = [r.get("acc5", float("nan")) for r in valid_mb_results]
            if os.getenv("ACC_FILL_NAN", "0") == "1":
                # 全 NaN 时 np.nanmean 会 warning；这里做保护
                acc_arr = np.array(acc_list, dtype=np.float32)
                valid = acc_arr[np.isfinite(acc_arr)]
                acc5 = float(valid.mean()) if valid.size > 0 else float("nan")
            else:
                acc5 = float(
                    np.mean([0.0 if (a is None or (isinstance(a, float) and np.isnan(a))) else a for a in acc_list]))
        else:
            acc5 = float("nan") if os.getenv("ACC_FILL_NAN", "0") == "1" else 0.0
        hot_ratio = float(np.mean([r.get("hot_ratio", 0.0) for r in valid_mb_results])) if valid_mb_results else 0.0

        # 4. 详细延迟指标 (带空值保护，修复 RuntimeWarning)
        # 修正 inv_net_ms: 只统计 > 0 的真实网络延迟
        net_lats = [r.get("net_lat", 0.0) for r in clean_invokes if r.get("net_lat", 0) > 0]
        inv_net_ms = float(np.mean(net_lats)) if net_lats else 0.0

        # 修正 inv_comp_ms: 只统计专家 FWD 且 < 10000ms 的真实计算
        raw_comp_lats = [
            r.get("comp_lat", 0.0) for r in clean_invokes
            if "expert_fwd" in r.get("func_name", "")
        ]
        clamped_comp_lats = [min(lat, 5000.0) for lat in raw_comp_lats if lat > 0]
        inv_comp_ms = float(np.mean(clamped_comp_lats)) if clamped_comp_lats else 0.0

        # 其他累加指标
        inv_queue_ms = float(np.sum([r.get("queue_lat", 0.0) for r in clean_invokes])) / max(1, mb)
        inv_cold_ms = float(np.sum([r.get("cold_penalty", 0.0) for r in clean_invokes])) / max(1, mb)
        inv_cold_cnt = int(np.sum([1 for r in clean_invokes if r.get("is_cold", False)]))

        # 5. 成本分类
        cost_pre = float(np.sum([r.get("cost_usd", 0.0) for r in clean_invokes if "pre_fwd" in r.get("func_name", "")]))
        cost_expert = float(
            np.sum([r.get("cost_usd", 0.0) for r in clean_invokes if "expert_fwd" in r.get("func_name", "")]))
        cost_post = float(
            np.sum([r.get("cost_usd", 0.0) for r in clean_invokes if "post_fwd" in r.get("func_name", "")]))
        total_cost = float(np.sum([r.get("cost_usd", 0.0) for r in clean_invokes]))

        # 6. SLO 计算
        step_time_ms = (time.perf_counter() - t_step0) * 1000.0

        # === [TAIL] update rolling + global tail/stability stats for step_time_ms ===
        split_key = split
        _STEP_TIME_ROLL.setdefault(split_key, deque(maxlen=TAIL_STAT_WINDOW))
        _STEP_TIME_ROLL[split_key].append(float(step_time_ms))
        roll_p95, roll_p99, roll_cv = _tail_stats_from_list(list(_STEP_TIME_ROLL[split_key]))

        # global: reservoir for percentiles + Welford for exact CV
        st = _STEP_TIME_WELFORD.setdefault(split_key, {"n": 0.0, "mean": 0.0, "m2": 0.0})
        _welford_update(st, float(step_time_ms))
        glob_cv = _welford_cv(st)
        seen = int(st["n"])
        _STEP_TIME_RES.setdefault(split_key, [])
        _reservoir_add(_STEP_TIME_RES[split_key], float(step_time_ms), cap=GLOBAL_STAT_RESERVOIR, seen=seen)
        glob_p95, glob_p99, _ = _tail_stats_from_list(_STEP_TIME_RES[split_key])

        DEADLINE_EST.update(step_time_ms)
        current_deadline = float(DEADLINE_EST.deadline_ms(step))
        # 违约判定：平均微批次时间是否超过 Deadline
        viol = 1.0 if (step_time_ms / mb) > current_deadline else 0.0

        # 7. 记录 Metrics
        metrics_record = StepMetrics(
            epoch=current_epoch,
            step=step,
            phase=split,
            loss=loss,
            acc_top5=acc5,
            step_time_ms=step_time_ms,
            step_time_p95_ms=roll_p95,
            step_time_p99_ms=roll_p99,
            step_time_cv=roll_cv,
            step_time_global_p95_ms=glob_p95,
            step_time_global_p99_ms=glob_p99,
            step_time_global_cv=glob_cv,
            predictor_r2=float(current_r2),
            predictor_mae=float(current_mae),
            inv_cold_cnt=inv_cold_cnt,
            inv_cold_ms=inv_cold_ms,
            inv_net_ms=inv_net_ms,
            inv_compute_ms=inv_comp_ms,
            inv_queue_ms=inv_queue_ms,
            cost_usd_step=total_cost,
            cost_usd_pre_fwd=cost_pre,
            cost_usd_post_fwd=cost_post,
            cost_usd_expert_fwd=cost_expert,
            hot_ratio=hot_ratio,
            deadline_ms=current_deadline,
            deadline_violation_frac=float(viol),
            ablation_mode=os.getenv("ABLATION_MODE", "full")
        )

        logger.log(metrics_record)

        if (step % LOG_TRAIN_EVERY) == 0 or split == "val":
            print(
                f"[{split}] step={step} loss={loss:.4f} acc5={acc5:.4f} "
                f"time={step_time_ms:.0f}ms p95={roll_p95:.0f}ms cv={roll_cv:.3f} cost=${total_cost:.6f} R2={current_r2:.3f} "
                f"comp={inv_comp_ms:.1f}ms net={inv_net_ms:.1f}ms", flush=True
            )


def main():
    # ==========================================
    # 【新增】初始化 LocalExecutor
    # ==========================================
    global LOCAL_EXECUTOR  # 1. 声明使用全局变量
    global logger
    logger = MetricsLogger(os.getenv("METRICS_FILE", "metrics.csv"))

    # 2. 如果开启了本地计算模式 (默认开启)，则初始化它
    if os.getenv("USE_HTTP_EXEC", "0") == "0":
        print(">>> [System] Initializing Global LocalExecutor (In-Process Mode)...")
        LOCAL_EXECUTOR = LocalExecutor()

    # 原有的启动逻辑
    asyncio.run(train())


if __name__ == "__main__": main()