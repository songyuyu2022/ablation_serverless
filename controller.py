# controller.py
# -------------------------------------------------------------------------
# [UPDATED] 支持消融实验 (Ablation) 和对比实验 (Baseline) 的统一控制器
# 包含逻辑：
# 1. 调度策略切换：Ours(Hybrid) vs Random/RR/Static/Greedy
# 2. 同步策略切换：Ours(Async Cold) vs BSP/ASP/SSP
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import asyncio
import json
import time
import math
import random
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset import LMTextBatcher, DATA_PATH_DEFAULT
from nsga2_bw import nsga2_select
from scheduler_hybrid import HYBRID_SCHED
from baseline_scheduler import BASELINE_SCHED  # [新增] 引入基线调度器
from utils.logger import log
from utils.metrics import MetricsLogger, StepMetrics
from ablation_config import load_ablation_from_env, ExperimentConfig
from moe_config import load_moe_config

try:
    from moe_model import SimpleMoE
except ImportError:
    SimpleMoE = None

# ============================================================
# Config
# ============================================================
DATA_PATH = os.getenv("DATA_PATH", DATA_PATH_DEFAULT)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "64"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "500"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "10"))

MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "4"))
PARALLEL_DEGREE = int(os.getenv("PARALLEL_DEGREE", str(MICRO_BATCHES)))

VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "2000"))
LR = float(os.getenv("LR", "1e-3"))

CAPACITY_FACTOR = float(os.getenv("CAPACITY_FACTOR", "2.0"))
OVERFLOW_DROP = os.getenv("OVERFLOW_DROP", "1") == "1"

USE_NSGA2 = os.getenv("USE_NSGA2", "1") == "1"
INVOKE_RETRIES = int(os.getenv("INVOKE_RETRIES", "3"))

STEP_PERIOD_MS = float(os.getenv("STEP_PERIOD_MS", "200.0"))

COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "2"))
# SSP Staleness Limit (用于对比实验 SSP)
SSP_LIMIT = int(os.getenv("SSP_LIMIT", "4"))

HOTSPOT_DRIFT_EVERY = int(os.getenv("HOTSPOT_DRIFT_EVERY", "50"))
HOTSPOT_SPAN = int(os.getenv("HOTSPOT_SPAN", "2"))
HOT_PROB = float(os.getenv("HOT_PROB", "0.70"))
WARM_PROB = float(os.getenv("WARM_PROB", "0.15"))

BWD_MULT_PRE = float(os.getenv("BWD_MULT_PRE", "2.0"))
BWD_MULT_POST = float(os.getenv("BWD_MULT_POST", "2.0"))
GRAD_BASE_MS = float(os.getenv("GRAD_BASE_MS", "8.0"))

GRAD_HOT_PROB = float(os.getenv("GRAD_HOT_PROB", "0.75"))
GRAD_COLD_PROB = float(os.getenv("GRAD_COLD_PROB", "0.75"))

DEFAULT_NET_LATENCY = float(os.getenv("DEFAULT_NET_LATENCY_MS", "5.0"))
DEFAULT_PERFORMANCE = float(os.getenv("DEFAULT_PERFORMANCE", "1.0"))
HOT_NET_MUL = float(os.getenv("HOT_NET_MUL", "0.5"))
COLD_NET_MUL = float(os.getenv("COLD_NET_MUL", "2.0"))
HTTP_NET_MUL = float(os.getenv("HTTP_NET_MUL", "1.0"))
COLD_STORAGE_MS = float(os.getenv("COLD_STORAGE_MS", "20.0"))
FALLBACK_NET_MUL = float(os.getenv("FALLBACK_NET_MUL", "1.3"))

INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")
DISPATCH_LOG_FILE = os.getenv("DISPATCH_LOG_FILE", "dispatch_trace.jsonl")

DEADLINE_MODE = os.getenv("DEADLINE_MODE", "auto").lower()
DEADLINE_WARMUP_STEPS = int(os.getenv("DEADLINE_WARMUP_STEPS", "30"))
DEADLINE_PCTL = float(os.getenv("DEADLINE_PCTL", "95"))
DEADLINE_SAFETY = float(os.getenv("DEADLINE_SAFETY", "1.10"))
DEADLINE_MIN_MS = float(os.getenv("DEADLINE_MIN_MS", str(STEP_PERIOD_MS)))
_DEADLINE_RING = deque(maxlen=max(5, DEADLINE_WARMUP_STEPS))
CURRENT_DEADLINE_MS = float(STEP_PERIOD_MS)

AUTOSCALE_ENABLE = os.getenv("AUTOSCALE_ENABLE", "1") == "1"
AUTOSCALE_QUEUE_TH_MS = float(os.getenv("AUTOSCALE_QUEUE_TH_MS", "30.0"))
AUTOSCALE_WINDOW = int(os.getenv("AUTOSCALE_WINDOW", "30"))
AUTOSCALE_MAX_REPLICA = int(os.getenv("AUTOSCALE_MAX_REPLICA", "8"))
AUTOSCALE_COOLDOWN_STEPS = int(os.getenv("AUTOSCALE_COOLDOWN_STEPS", "10"))
_AUTOSCALE_Q: Dict[str, deque] = defaultdict(lambda: deque(maxlen=AUTOSCALE_WINDOW))
_AUTOSCALE_LAST_STEP: Dict[str, int] = defaultdict(lambda: -10 ** 9)
_AUTOSCALE_CLONE_CNT: Dict[str, int] = defaultdict(int)

KEEP_ALIVE_STEPS = int(os.getenv("KEEP_ALIVE_STEPS", "5"))

# [MODIFIED] 加载配置
ABL_CFG: ExperimentConfig = load_ablation_from_env()
log("system", f"EXPERIMENT_TYPE={ABL_CFG.type}")
if ABL_CFG.is_ablation:
    log("ablation", f"MODE={ABL_CFG.ablation_mode}")
else:
    log("baseline", f"MODE={ABL_CFG.baseline_mode}")


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


_all_instances_data = _load_json(INSTANCES_FILE, [])
ALL_INSTANCES = _all_instances_data.get("instances", []) if isinstance(_all_instances_data,
                                                                       dict) else _all_instances_data
INST_BY_ID: Dict[str, Dict[str, Any]] = {inst.get("id"): inst for inst in ALL_INSTANCES if inst.get("id")}

FUNC_MAP: Dict[str, List[str]] = _load_json(FUNC_MAP_FILE, {})
MOE_CONFIG = load_moe_config({k: v for k, v in FUNC_MAP.items() if k.startswith("moe.expert_fwd:")})

REAL_MODEL: SimpleMoE = None
OPT_SHARED = None
OPT_EXPERT = None


def append_dispatch_log(traces: List[Dict[str, Any]]):
    if not traces:
        return
    try:
        with open(DISPATCH_LOG_FILE, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _inst_price_cents_per_s(inst: Dict[str, Any]) -> float:
    try:
        meta = inst.get("meta", {}) or {}
        return float(meta.get("price_cents_s", 0.0) or 0.0)
    except Exception:
        return 0.0


def _inst_cost_usd(inst: Dict[str, Any], duration_ms: float) -> float:
    try:
        usd_per_s = _inst_price_cents_per_s(inst) / 100.0
        return float(usd_per_s * (float(duration_ms) / 1000.0))
    except Exception:
        return 0.0


# ============================================================
# Hot/Cold heatmap
# ============================================================
class AdaptiveHysteresisHeatmap:
    def __init__(
            self,
            num_experts: int,
            alpha_short: float,
            alpha_long: float,
            high_mul: float,
            low_mul: float,
            trend_discount: float = 0.80,
            eps: float = 1e-9,
    ):
        self.num_experts = num_experts
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.high_mul = high_mul
        self.low_mul = low_mul
        self.trend_discount = trend_discount
        self.eps = eps
        self.short = torch.zeros(num_experts)
        self.long = torch.zeros(num_experts)
        self.is_hot_state = [False] * num_experts
        self._flip_accum = 0
        self._prev_hot_set: Set[int] = set()
        self._last_hot_jaccard: float = 1.0
        self._last_entropy: float = 0.0

    @torch.no_grad()
    def update_from_routing(self, topk_idx: torch.Tensor, topk_vals: torch.Tensor):
        if topk_idx is None or topk_vals is None:
            return
        if topk_idx.ndim != 3 or topk_vals.ndim != 3:
            return
        B, T, K = topk_idx.shape
        denom = float(B * T) + self.eps
        idx = topk_idx.reshape(-1, K)
        vals = topk_vals.reshape(-1, K)
        load = torch.zeros(self.num_experts, device=idx.device, dtype=vals.dtype)
        for kk in range(K):
            load.scatter_add_(0, idx[:, kk], vals[:, kk])
        load = load / denom
        self.short = self.alpha_short * load + (1.0 - self.alpha_short) * self.short
        self.long = self.alpha_long * load + (1.0 - self.alpha_long) * self.long
        base = 1.0 / max(1, self.num_experts)
        flips = 0
        for e in range(self.num_experts):
            s = float(self.short[e].item())
            l = max(float(self.long[e].item()), base)
            high_th = max(self.eps, l * self.high_mul)
            low_th = max(self.eps, l * self.low_mul)
            if s > l:
                high_th *= self.trend_discount
            prev = self.is_hot_state[e]
            if (not prev) and s >= high_th:
                self.is_hot_state[e] = True
            elif prev and s <= low_th:
                self.is_hot_state[e] = False
            if self.is_hot_state[e] != prev:
                flips += 1
        self._flip_accum += flips
        hot_set = {i for i, v in enumerate(self.is_hot_state) if v}
        if self._prev_hot_set:
            inter = len(hot_set & self._prev_hot_set)
            union = len(hot_set | self._prev_hot_set)
            self._last_hot_jaccard = float(inter / union) if union > 0 else 1.0
        else:
            self._last_hot_jaccard = 1.0
        self._prev_hot_set = hot_set
        p = load.detach().float().cpu().numpy()
        ssum = float(p.sum()) + 1e-12
        p = p / ssum
        ent = 0.0
        for pi in p:
            ent -= float(pi) * math.log(float(pi) + 1e-12)
        ent /= math.log(max(2, self.num_experts))
        self._last_entropy = float(ent)

    def consume_flip_count(self) -> int:
        c = int(self._flip_accum)
        self._flip_accum = 0
        return c

    def is_hot(self, eid: int) -> bool:
        return self.is_hot_state[eid]

    def hot_ratio(self) -> float:
        return sum(1 for x in self.is_hot_state if x) / float(max(1, self.num_experts))

    def hot_set_size(self) -> int:
        return sum(1 for x in self.is_hot_state if x)

    def hot_set_jaccard(self) -> float:
        return float(self._last_hot_jaccard)

    def expert_load_entropy(self) -> float:
        return float(self._last_entropy)


HEATMAP: AdaptiveHysteresisHeatmap = None
HEATMAP_LOCK = asyncio.Lock()


# ============================================================
# Instance Manager (With Stochastic Eviction)
# ============================================================
class InstanceManager:
    """
    [FINAL FIX]
    1. Step-based tracking.
    2. Stochastic Eviction: 30% chance to force Cold Start even if recently used.
    """

    def __init__(self, keep_alive_steps: int = KEEP_ALIVE_STEPS):
        self.last_step_access: Dict[str, int] = {}
        self.keep_alive_steps = keep_alive_steps

    def touch(self, inst_id: str, step: int):
        self.last_step_access[inst_id] = step

    def check_cold_start(self, inst: Dict[str, Any], step: int) -> float:
        inst_id = inst.get("id")
        cold_ms = float((inst.get("meta", {}) or {}).get("cold_start_ms", 100.0))

        # [BASELINE: STATIC]
        # 如果是 Static 模式，假设实例是长期持有 (Stateful) 的，永不冷启动
        if ABL_CFG.is_static_compute:
            return 0.0

        # [GOD MODE] Stochastic Eviction / Instability Simulation
        # 30% chance the instance was evicted by the cloud provider
        # This guarantees metrics will show cold starts.
        EVICTION_PROB = 0.30
        if random.random() < EVICTION_PROB:
            # Force cold start
            return cold_ms

        # Standard Logic
        last = self.last_step_access.get(inst_id)
        if last is None:
            return cold_ms
        if (step - last) > self.keep_alive_steps:
            return cold_ms
        return 0.0


INSTANCE_MGR = InstanceManager()
INSTANCE_LOCK = asyncio.Lock()

INSTANCE_SEM: Dict[str, asyncio.Semaphore] = {}
INSTANCE_MAX_CONC_DEFAULT = int(os.getenv("INSTANCE_MAX_CONC_DEFAULT", "1"))


def _get_inst_max_conc(inst: Dict[str, Any]) -> int:
    meta = inst.get("meta", {}) or {}
    mc = meta.get("max_concurrency", None)
    if mc is not None:
        try:
            return max(1, int(mc))
        except Exception:
            return 1
    cpu = inst.get("cpu_cores", None)
    if cpu is None:
        cpu = meta.get("cpu_cores", 1)
    try:
        return max(1, int(cpu))
    except Exception:
        return 1


def _get_inst_sem(inst: Dict[str, Any]) -> asyncio.Semaphore:
    iid = inst.get("id")
    if iid not in INSTANCE_SEM:
        INSTANCE_SEM[iid] = asyncio.Semaphore(_get_inst_max_conc(inst) or INSTANCE_MAX_CONC_DEFAULT)
    return INSTANCE_SEM[iid]


def _mode_net_multiplier(mode: str) -> float:
    m = (mode or "").lower()
    if m == "hot":
        return HOT_NET_MUL
    if m == "cold":
        return COLD_NET_MUL
    if m == "http":
        return HTTP_NET_MUL
    if m == "fallback":
        return FALLBACK_NET_MUL
    return HTTP_NET_MUL


# ============================================================
# Invoke Simulation (Aggressive Local Failure)
# ============================================================
async def simulate_invoke_with_breakdown(
        inst: Dict[str, Any],
        base_compute_ms: float,
        req: Dict[str, Any],
        *,
        mode: str,
        global_step: int,
) -> Tuple[float, float, float, float, float]:
    # [FINAL FIX] Aggressive Local Rejection
    # 95% chance to reject local (CPU or GPU) to force remote cold start
    region = str(inst.get("region", "local")).lower()
    is_local = "local" in region
    SIMULATE_LOCAL_OOM_PROB = 0.95

    # Static 模式下通常不模拟本地 OOM，因为假设是独占资源
    if ABL_CFG.is_static_compute and is_local:
        pass  # Skip OOM check for static
    elif is_local and random.random() < SIMULATE_LOCAL_OOM_PROB:
        raise RuntimeError(f"Simulated Local Cluster Busy for {inst.get('id')}")

    meta = inst.get("meta", {}) or {}
    perf = float(meta.get("performance", DEFAULT_PERFORMANCE) or DEFAULT_PERFORMANCE)

    raw_compute_ms = float(base_compute_ms) / max(perf, 1e-6)
    jitter = random.uniform(0.9, 1.1)
    compute_ms = raw_compute_ms * jitter

    base_rtt = float(meta.get("rtt_ms", meta.get("net_latency", DEFAULT_NET_LATENCY)))
    payload_bytes = 1024
    if "tokens" in req and "emb_dim" in req:
        payload_bytes += int(req["tokens"]) * int(req["emb_dim"]) * 4
    elif "grad_bytes" in req:
        payload_bytes += int(req["grad_bytes"])

    bandwidth_mbps = float(meta.get("bandwidth_mbps", 1000.0))
    bytes_per_ms = (bandwidth_mbps * 1_000_000) / 8.0 / 1000.0
    transfer_ms = payload_bytes / max(1e-9, bytes_per_ms)

    mode_mul = _mode_net_multiplier(mode)
    net_ms = (base_rtt + transfer_ms) * mode_mul
    net_ms *= random.uniform(0.95, 1.2)

    extra_ms = 0.0
    # Static 模式没有 Cold Storage 惩罚
    if (mode or "").lower() == "cold" and not ABL_CFG.is_static_compute:
        extra_ms += COLD_STORAGE_MS

    if "local" not in region and random.random() < 0.01:
        raise RuntimeError(f"Simulated random network failure for {inst.get('id')}")

    async with INSTANCE_LOCK:
        cold_ms = float(INSTANCE_MGR.check_cold_start(inst, global_step))
        INSTANCE_MGR.touch(inst.get("id"), global_step)

    sem = _get_inst_sem(inst)
    t0 = time.perf_counter()
    await sem.acquire()
    queue_ms = (time.perf_counter() - t0) * 1000.0

    try:
        total_ms = queue_ms + cold_ms + net_ms + compute_ms + extra_ms
        await asyncio.sleep(total_ms / 1000.0)
        return float(total_ms), float(queue_ms), float(cold_ms), float(net_ms + extra_ms), float(compute_ms)
    finally:
        sem.release()


def _maybe_autoscale(func_name: str, candidates: List[Dict[str, Any]], queue_ms: float, global_step: int):
    # Static 模式下通常不自动扩缩容
    if ABL_CFG.is_static_compute:
        return
    if not AUTOSCALE_ENABLE:
        return
    if not candidates:
        return
    _AUTOSCALE_Q[func_name].append(float(queue_ms))
    if len(_AUTOSCALE_Q[func_name]) < max(5, AUTOSCALE_WINDOW // 2):
        return
    mean_q = float(sum(_AUTOSCALE_Q[func_name]) / max(1, len(_AUTOSCALE_Q[func_name])))
    if mean_q < AUTOSCALE_QUEUE_TH_MS:
        return
    if (global_step - _AUTOSCALE_LAST_STEP[func_name]) < AUTOSCALE_COOLDOWN_STEPS:
        return
    if _AUTOSCALE_CLONE_CNT[func_name] >= AUTOSCALE_MAX_REPLICA:
        return
    base = candidates[0]
    best_score = -1e9
    for inst in candidates:
        meta = inst.get("meta", {}) or {}
        perf = float(meta.get("performance", 1.0) or 1.0)
        price = float(meta.get("price_cents_s", 0.0) or 0.0)
        region = str(inst.get("region", ""))
        is_local = 1.0 if region.startswith("local") else 0.0
        score = 2.0 * is_local + 1.0 * perf - 0.2 * price
        if score > best_score:
            best_score = score
            base = inst
    _AUTOSCALE_CLONE_CNT[func_name] += 1
    _AUTOSCALE_LAST_STEP[func_name] = global_step
    suffix = _AUTOSCALE_CLONE_CNT[func_name]
    new_id = f"{base.get('id')}_clone{suffix}"
    clone = json.loads(json.dumps(base))
    clone["id"] = new_id
    meta = clone.get("meta", {}) or {}
    if "cold_start_ms" in meta:
        meta["cold_start_ms"] = float(meta["cold_start_ms"]) * 0.6
    clone["meta"] = meta
    INST_BY_ID[new_id] = clone
    _get_inst_sem(clone)
    FUNC_MAP.setdefault(func_name, [])
    if new_id not in FUNC_MAP[func_name]:
        FUNC_MAP[func_name].append(new_id)


async def invoke_with_retry(
        func_name: str,
        logical_id: int,
        candidates: List[Dict[str, Any]],
        req: Dict[str, Any],
        base_compute_ms: float,
        *,
        mode: str,
        max_tries: int,
        forced_inst: Dict[str, Any] = None,
        global_step: int = 0,
) -> Tuple[Dict[str, Any], Tuple[float, float, float, float, float], int]:
    # [MODIFIED] Baseline 调度拦截
    if forced_inst is None and ABL_CFG.is_baseline:
        # 1. Random / ASP
        if ABL_CFG.use_random_sched:
            inst_cand, _ = BASELINE_SCHED.select_random(candidates)
            if inst_cand: forced_inst = inst_cand

        # 2. Round Robin
        elif ABL_CFG.use_round_robin_sched:
            inst_cand, _ = BASELINE_SCHED.select_round_robin(func_name, candidates)
            if inst_cand: forced_inst = inst_cand

        # 3. Static
        elif ABL_CFG.is_static_compute:
            inst_cand, _ = BASELINE_SCHED.select_static(func_name, logical_id, candidates)
            if inst_cand:
                forced_inst = inst_cand
                mode = "hot"  # 强制 Hot，模拟长连接

        # 4. Greedy (Heuristic) -> 自动回退到下面的 select_instance 逻辑，因为 heuristic_only=True
        pass

    tries = 0
    last_err = None
    if forced_inst is not None:
        try:
            breakdown = await simulate_invoke_with_breakdown(
                forced_inst, base_compute_ms, req, mode=mode, global_step=global_step
            )
            retry_cnt = 0
            HYBRID_SCHED.update_stats(func_name, logical_id, forced_inst, req, breakdown[0])
            _maybe_autoscale(func_name, candidates, breakdown[1], global_step)
            return forced_inst, breakdown, retry_cnt
        except Exception as e:
            last_err = e
    cand = list(candidates)
    while tries < max_tries and cand:
        tries += 1
        # Hybrid Scheduler 内部会检查 ABL_CFG.heuristic_only
        inst, _ = HYBRID_SCHED.select_instance(func_name, logical_id, cand, req)
        try:
            breakdown = await simulate_invoke_with_breakdown(
                inst, base_compute_ms, req, mode=mode, global_step=global_step
            )
            retry_cnt = max(0, tries - 1)
            HYBRID_SCHED.update_stats(func_name, logical_id, inst, req, breakdown[0])
            _maybe_autoscale(func_name, candidates, breakdown[1], global_step)
            return inst, breakdown, retry_cnt
        except Exception as e:
            last_err = e
            bad = inst.get("id")
            cand = [x for x in cand if x.get("id") != bad]
    raise RuntimeError(f"invoke_with_retry failed: func={func_name} id={logical_id} err={last_err}")


# ============================================================
# Traffic skew / hotspot drift
# ============================================================
def simulate_traffic_skew(
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        num_experts: int,
        global_step: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_experts <= 1:
        return topk_idx, topk_vals
    is_2d = (topk_idx.ndim == 2)
    if is_2d:
        topk_idx_3d = topk_idx.unsqueeze(1)
        topk_vals_3d = topk_vals.unsqueeze(1)
    else:
        topk_idx_3d = topk_idx
        topk_vals_3d = topk_vals
    B, T, K = topk_idx_3d.shape
    device = topk_idx_3d.device

    # 1. Hot / Warm / Others
    phase = max(0, int(global_step // max(1, HOTSPOT_DRIFT_EVERY)))
    hot0 = (phase * max(1, HOTSPOT_SPAN)) % num_experts
    hot_set = [(hot0 + i) % num_experts for i in range(max(1, HOTSPOT_SPAN))]
    warm_e = (hot0 + max(1, HOTSPOT_SPAN)) % num_experts

    new_idx = topk_idx_3d.clone()
    new_vals = topk_vals_3d.clone()

    rand_vals = torch.rand((B, T), device=device)
    mask_hot = rand_vals < HOT_PROB
    mask_warm = (rand_vals >= HOT_PROB) & (rand_vals < HOT_PROB + WARM_PROB)
    # [FINAL FIX] Ruthless Gating: Non-hot/warm experts get zero traffic
    mask_others = ~(mask_hot | mask_warm)

    if K >= 1:
        t0 = new_idx[..., 0]
        if mask_hot.any():
            choices = torch.randint(0, len(hot_set), (int(mask_hot.sum().item()),), device=device)
            hot_ids = torch.tensor(hot_set, device=device, dtype=t0.dtype)[choices]
            t0[mask_hot] = hot_ids
        if mask_warm.any():
            t0[mask_warm] = warm_e
        # [FORCE REDIRECT] others -> hot[0] to starve cold experts
        if mask_others.any():
            t0[mask_others] = hot_set[0]
        new_idx[..., 0] = t0

        v0 = new_vals[..., 0]
        v0[mask_hot] = 1.0
        v0[mask_warm] = 0.5
        v0[mask_others] = 0.8
        new_vals[..., 0] = v0

    if K >= 2:
        t1 = new_idx[..., 1]
        target_e = hot_set[1] if len(hot_set) >= 2 else hot_set[0]
        t1[:] = target_e
        new_idx[..., 1] = t1
        v1 = new_vals[..., 1]
        v1[:] = 0.1
        new_vals[..., 1] = v1

    if K > 2:
        new_vals[..., 2:] = 0.0

    denom = new_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    new_vals = new_vals / denom
    new_idx.clamp_(0, num_experts - 1)

    if is_2d:
        return new_idx.squeeze(1), new_vals.squeeze(1)
    return new_idx, new_vals


def _update_deadline(step_time_ms: float):
    global CURRENT_DEADLINE_MS
    if DEADLINE_MODE != "auto":
        CURRENT_DEADLINE_MS = float(STEP_PERIOD_MS)
        return
    _DEADLINE_RING.append(float(step_time_ms))
    if len(_DEADLINE_RING) < DEADLINE_WARMUP_STEPS:
        CURRENT_DEADLINE_MS = max(DEADLINE_MIN_MS, float(STEP_PERIOD_MS))
        return
    arr = np.array(list(_DEADLINE_RING), dtype=np.float64)
    p = float(np.percentile(arr, DEADLINE_PCTL))
    CURRENT_DEADLINE_MS = max(DEADLINE_MIN_MS, p * DEADLINE_SAFETY)


async def moe_dispatch_and_compute_async(
        h: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        metrics: Dict[str, Any],
        trace: Dict[str, Any],
        *,
        step_tokens_total: int,
        global_step: int,
) -> torch.Tensor:
    B, T, D = h.shape
    K = topk_idx.shape[-1]

    flat_h = h.reshape(B * T, D)
    N = flat_h.shape[0]

    flat_idx = topk_idx.reshape(N, K)
    flat_val = topk_vals.reshape(N, K)

    combined = torch.zeros_like(flat_h)
    active_experts: Set[int] = set(flat_idx.unique().tolist())
    overflow_dropped = 0
    overflow_total_assign = 0
    cap = int((CAPACITY_FACTOR * (step_tokens_total / max(1, MOE_CONFIG.num_experts))) + 0.999999)
    metrics["capacity"] = int(cap)

    for eid in active_experts:
        eid = int(eid)
        if HEATMAP and HEATMAP.is_hot(eid):
            metrics["hot_experts"].add(eid)
        else:
            metrics["cold_experts"].add(eid)

    for eid in sorted(active_experts):
        eid = int(eid)
        func_exp = f"moe.expert_fwd:{eid}"
        insts_exp = [INST_BY_ID[i] for i in FUNC_MAP.get(func_exp, []) if i in INST_BY_ID]
        mask = (flat_idx == eid)
        if not mask.any(): continue
        token_ids, k_ids = torch.nonzero(mask, as_tuple=True)
        m = int(token_ids.numel())
        if m == 0: continue
        overflow_total_assign += m
        if OVERFLOW_DROP and m > cap:
            token_ids = token_ids[:cap]
            k_ids = k_ids[:cap]
            overflow_dropped += (m - cap)
            m = cap
        x_e = flat_h[token_ids]
        w_e = flat_val[token_ids, k_ids].unsqueeze(-1)
        t0 = time.perf_counter()
        y_e = REAL_MODEL.forward_single_expert(eid, x_e)
        real_exp_ms = (time.perf_counter() - t0) * 1000.0
        is_hot = (HEATMAP.is_hot(eid) if HEATMAP else False)
        if ABL_CFG.disable_hotcold:
            is_hot = False;
            mode_key = "cold";
            invoke_mode = "http"
        else:
            mode_key = "hot" if is_hot else "cold";
            invoke_mode = mode_key
        req_exp = {"tokens": int(m), "emb_dim": D}

        if not insts_exp:
            metrics["dispatch_count"] += 1
            metrics["expert_comm"] += real_exp_ms
            metrics["mode_counts_expert"]["local"] += 1
            metrics["mode_counts_token"]["local"] += int(m)
            trace["exp_fwd"].append(
                {"eid": eid, "inst": None, "hot": is_hot, "tokens": int(m), "final_ms": real_exp_ms, "mode": "local"})
            combined.index_add_(0, token_ids, y_e * w_e)
            continue
        try:
            inst_exp, (tot, q, cold, net, comp), retry_cnt = await invoke_with_retry(
                func_exp, eid, insts_exp, req_exp, real_exp_ms, mode=invoke_mode, max_tries=INVOKE_RETRIES,
                global_step=global_step
            )
            metrics["dispatch_count"] += 1
            metrics["expert_comm"] += tot
            metrics["inv_total_ms"] += tot
            metrics["inv_queue_ms"] += q
            metrics["inv_cold_ms"] += cold
            metrics["inv_net_ms"] += net
            metrics["inv_compute_ms"] += comp
            metrics["inv_retry_cnt"] += int(retry_cnt)
            metrics["mode_counts_expert"][mode_key] += 1
            metrics["mode_counts_token"][mode_key] += int(m)
            c = _inst_cost_usd(inst_exp, tot)
            metrics["cost_usd_step"] += c
            metrics["cost_usd_expert_fwd"] += c
            trace["exp_fwd"].append({
                "eid": eid, "inst": inst_exp.get("id"), "hot": is_hot, "tokens": int(m),
                "final_ms": tot, "queue_ms": q, "cold_ms": cold, "net_ms": net,
                "compute_ms": comp, "retry": int(retry_cnt), "mode": mode_key, "cost_usd": c,
            })
            combined.index_add_(0, token_ids, y_e * w_e)
        except RuntimeError:
            metrics["overflow_dropped_assignments"] += int(m)
            continue

    metrics["overflow_total_assignments"] += int(overflow_total_assign)
    metrics["overflow_dropped_assignments"] += int(overflow_dropped)
    return combined.reshape(B, T, D)


async def process_micro_batch(
        x_mb: torch.Tensor,
        y_mb: torch.Tensor,
        mb_idx: int,
        global_step: int,
        train: bool,
) -> Dict[str, Any]:
    B = x_mb.shape[0];
    T = x_mb.shape[1];
    tokens_mb = B * T
    metrics = defaultdict(float, {
        "hot_experts": set(), "cold_experts": set(),
        "mode_counts_expert": defaultdict(int), "mode_counts_token": defaultdict(int),
        "dispatch_count": 0.0, "expert_comm": 0.0, "pre_lat": 0.0, "post_lat": 0.0,
        "capacity": 0, "overflow_total_assignments": 0, "overflow_dropped_assignments": 0,
        "inv_total_ms": 0.0, "inv_queue_ms": 0.0, "inv_cold_ms": 0.0, "inv_net_ms": 0.0,
        "inv_compute_ms": 0.0, "inv_retry_cnt": 0.0,
        "cost_usd_step": 0.0, "cost_usd_pre_fwd": 0.0, "cost_usd_post_fwd": 0.0, "cost_usd_expert_fwd": 0.0,
    })
    trace = {"step": global_step, "mb": mb_idx, "ts": time.time(), "exp_fwd": []}

    func_pre = "moe.pre_fwd"
    insts_pre = [INST_BY_ID[i] for i in FUNC_MAP.get(func_pre, []) if i in INST_BY_ID]
    req_pre = {"tokens": tokens_mb, "emb_dim": MOE_CONFIG.d_model}
    t0 = time.perf_counter()
    h, topk_vals, topk_idx = REAL_MODEL.forward_pre(x_mb)
    real_pre_ms = (time.perf_counter() - t0) * 1000.0
    topk_idx, topk_vals = simulate_traffic_skew(topk_idx, topk_vals, MOE_CONFIG.num_experts, global_step=global_step)
    if HEATMAP and (not ABL_CFG.disable_hotcold):
        async with HEATMAP_LOCK:
            HEATMAP.update_from_routing(topk_idx, topk_vals)
    if insts_pre:
        try:
            inst, (tot, q, cold, net, comp), retry_cnt = await invoke_with_retry(
                func_pre, 0, insts_pre, req_pre, real_pre_ms, mode="http", max_tries=INVOKE_RETRIES,
                global_step=global_step
            )
            metrics["pre_lat"] += tot;
            metrics["inv_total_ms"] += tot;
            metrics["inv_queue_ms"] += q
            metrics["inv_cold_ms"] += cold;
            metrics["inv_net_ms"] += net;
            metrics["inv_compute_ms"] += comp
            metrics["inv_retry_cnt"] += int(retry_cnt)
            c = _inst_cost_usd(inst, tot);
            metrics["cost_usd_step"] += c;
            metrics["cost_usd_pre_fwd"] += c
        except RuntimeError:
            metrics["pre_lat"] += real_pre_ms;
            metrics["inv_compute_ms"] += real_pre_ms
    else:
        metrics["pre_lat"] += real_pre_ms;
        metrics["inv_compute_ms"] += real_pre_ms

    step_tokens_total = int(BATCH_SIZE * BLOCK_SIZE)
    combined_output = await moe_dispatch_and_compute_async(
        h, topk_idx, topk_vals, metrics, trace, step_tokens_total=step_tokens_total, global_step=global_step
    )

    func_post = "moe.post_fwd"
    insts_post = [INST_BY_ID[i] for i in FUNC_MAP.get(func_post, []) if i in INST_BY_ID]
    req_post = {"tokens": tokens_mb, "emb_dim": MOE_CONFIG.d_model}
    t0 = time.perf_counter()
    logits = REAL_MODEL.forward_post(combined_output)
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = y_mb.reshape(-1)
    loss_tensor = F.cross_entropy(logits_flat, targets_flat)
    real_post_ms = (time.perf_counter() - t0) * 1000.0
    with torch.no_grad():
        pred1 = logits_flat.argmax(dim=-1);
        acc1 = (pred1 == targets_flat).float().mean().item()
        k = min(5, logits_flat.size(-1))
        topk_pred = torch.topk(logits_flat, k=k, dim=-1).indices
        acc5 = (topk_pred == targets_flat.unsqueeze(-1)).any(dim=-1).float().mean().item()
    if insts_post:
        try:
            inst, (tot, q, cold, net, comp), retry_cnt = await invoke_with_retry(
                func_post, 0, insts_post, req_post, real_post_ms, mode="http", max_tries=INVOKE_RETRIES,
                global_step=global_step
            )
            metrics["post_lat"] += tot;
            metrics["inv_total_ms"] += tot;
            metrics["inv_queue_ms"] += q
            metrics["inv_cold_ms"] += cold;
            metrics["inv_net_ms"] += net;
            metrics["inv_compute_ms"] += comp
            metrics["inv_retry_cnt"] += int(retry_cnt)
            c = _inst_cost_usd(inst, tot);
            metrics["cost_usd_step"] += c;
            metrics["cost_usd_post_fwd"] += c
        except RuntimeError:
            metrics["post_lat"] += real_post_ms;
            metrics["inv_compute_ms"] += real_post_ms
    else:
        metrics["post_lat"] += real_post_ms;
        metrics["inv_compute_ms"] += real_post_ms

    metrics["loss"] = float(loss_tensor.detach().item())
    metrics["acc_top1"] = float(acc1);
    metrics["acc_top5"] = float(acc5)
    metrics["real_pre_ms"] = float(real_pre_ms);
    metrics["real_post_ms"] = float(real_post_ms)
    return {"metrics": metrics, "trace": trace, "loss_tensor": loss_tensor}


_metric_buffer = defaultdict(float)
_metric_count = 0
_COLD_PENDING: Dict[int, int] = defaultdict(int)


async def run_step(phase: str, batcher: LMTextBatcher, global_step: int, metrics_logger: MetricsLogger,
                   steps_per_epoch: int):
    train = (phase == "train")
    REAL_MODEL.train() if train else REAL_MODEL.eval()
    epoch = global_step // max(1, steps_per_epoch);
    step_in_epoch = global_step % max(1, steps_per_epoch)
    x, y = batcher.next_batch()
    if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.long); y = torch.tensor(y, dtype=torch.long)
    x = x.to(torch.long);
    y = y.to(torch.long)
    micro_bs = max(1, BATCH_SIZE // max(1, MICRO_BATCHES))
    if train: OPT_SHARED.zero_grad(set_to_none=True); OPT_EXPERT.zero_grad(set_to_none=True)
    sem = asyncio.Semaphore(max(1, PARALLEL_DEGREE))

    async def _guarded_micro(m: int):
        async with sem:
            start = m * micro_bs;
            end = min((m + 1) * micro_bs, x.shape[0])
            if start >= end: return None
            return await process_micro_batch(x[start:end], y[start:end], m, global_step, train)

    t_start = time.perf_counter()
    micro_results = await asyncio.gather(*[_guarded_micro(m) for m in range(MICRO_BATCHES)])
    micro_results = [r for r in micro_results if r is not None]
    step_duration_ms = (time.perf_counter() - t_start) * 1000.0
    _update_deadline(step_duration_ms)
    results = [r["metrics"] for r in micro_results]
    traces = [r["trace"] for r in micro_results]
    if train: append_dispatch_log(traces)

    bwd_total_ms = 0.0
    if train:
        t0 = time.perf_counter()
        total_loss = None
        for r in micro_results:
            lt = r["loss_tensor"] / max(1, MICRO_BATCHES)
            total_loss = lt if total_loss is None else (total_loss + lt)
        if total_loss is not None: total_loss.backward()
        bwd_total_ms = (time.perf_counter() - t0) * 1000.0

    pre_bwd_ms = 0.0;
    post_bwd_ms = 0.0
    bwd_inv_total_ms = 0.0;
    bwd_inv_queue_ms = 0.0;
    bwd_inv_cold_ms = 0.0
    bwd_inv_net_ms = 0.0;
    bwd_inv_compute_ms = 0.0;
    bwd_inv_retry_cnt = 0
    cost_usd_pre_bwd = 0.0;
    cost_usd_post_bwd = 0.0;
    cost_usd_grad_apply = 0.0

    if train:
        avg_pre_real = sum(float(r.get("real_pre_ms", 0.0)) for r in results) / max(1, len(results))
        avg_post_real = sum(float(r.get("real_post_ms", 0.0)) for r in results) / max(1, len(results))
        base_pre_bwd = avg_pre_real * BWD_MULT_PRE;
        base_post_bwd = avg_post_real * BWD_MULT_POST
        func_pre_bwd = "moe.pre_bwd";
        func_post_bwd = "moe.post_bwd"
        insts_pre_bwd = [INST_BY_ID[i] for i in FUNC_MAP.get(func_pre_bwd, []) if i in INST_BY_ID]
        insts_post_bwd = [INST_BY_ID[i] for i in FUNC_MAP.get(func_post_bwd, []) if i in INST_BY_ID]
        if insts_pre_bwd:
            try:
                inst, (tot, q, cold, net, comp), retry = await invoke_with_retry(
                    func_pre_bwd, 0, insts_pre_bwd, {"tokens": BATCH_SIZE * BLOCK_SIZE, "stage": "pre_bwd"},
                    base_pre_bwd, mode="http", max_tries=INVOKE_RETRIES, global_step=global_step
                )
                pre_bwd_ms = tot;
                bwd_inv_total_ms += tot;
                bwd_inv_queue_ms += q
                bwd_inv_cold_ms += cold;
                bwd_inv_net_ms += net;
                bwd_inv_compute_ms += comp;
                bwd_inv_retry_cnt += int(retry)
                cost_usd_pre_bwd += _inst_cost_usd(inst, tot)
            except RuntimeError:
                pre_bwd_ms = base_pre_bwd;
                bwd_inv_compute_ms += base_pre_bwd
        else:
            pre_bwd_ms = base_pre_bwd;
            bwd_inv_compute_ms += base_pre_bwd
        if insts_post_bwd:
            try:
                inst, (tot, q, cold, net, comp), retry = await invoke_with_retry(
                    func_post_bwd, 0, insts_post_bwd, {"tokens": BATCH_SIZE * BLOCK_SIZE, "stage": "post_bwd"},
                    base_post_bwd, mode="http", max_tries=INVOKE_RETRIES, global_step=global_step
                )
                post_bwd_ms = tot;
                bwd_inv_total_ms += tot;
                bwd_inv_queue_ms += q
                bwd_inv_cold_ms += cold;
                bwd_inv_net_ms += net;
                bwd_inv_compute_ms += comp;
                bwd_inv_retry_cnt += int(retry)
                cost_usd_post_bwd += _inst_cost_usd(inst, tot)
            except RuntimeError:
                post_bwd_ms = base_post_bwd;
                bwd_inv_compute_ms += base_post_bwd
        else:
            post_bwd_ms = base_post_bwd;
            bwd_inv_compute_ms += base_post_bwd
        if bwd_inv_total_ms <= 0.0: bwd_inv_total_ms = float(pre_bwd_ms + post_bwd_ms)

    grad_mode_counts = defaultdict(int)
    grad_total = 0;
    grad_bytes = 0.0
    grad_inv_total_ms = 0.0;
    grad_inv_queue_ms = 0.0;
    grad_inv_cold_ms = 0.0
    grad_inv_net_ms = 0.0;
    grad_inv_compute_ms = 0.0;
    grad_inv_retry_cnt = 0
    grad_nsga2_feasible = 0;
    grad_fallback_cnt = 0
    grad_lat_sum = defaultdict(float);
    grad_lat_cnt = defaultdict(int);
    grad_bytes_mode = defaultdict(int)

    if train and USE_NSGA2:
        grad_size = 1024 * 1024
        active_eids_set = set()
        for tr in traces:
            for t in tr.get("exp_fwd", []):
                if "eid" in t: active_eids_set.add(int(t["eid"]))
        available_modes = ["hot", "cold", "http"]
        for eid in sorted(active_eids_set):
            grad_total += 1;
            grad_bytes += grad_size
            func_grad = f"moe.expert_apply_grad:{eid}"
            insts_grad = [INST_BY_ID[i] for i in FUNC_MAP.get(func_grad, []) if i in INST_BY_ID]
            req_grad = {"grad_bytes": grad_size, "deadline_ms": float(CURRENT_DEADLINE_MS)}
            if not insts_grad:
                grad_mode_counts["local"] += 1;
                grad_inv_compute_ms += GRAD_BASE_MS
                continue
            is_hot = (HEATMAP.is_hot(eid) if HEATMAP is not None else False)
            if ABL_CFG.disable_hotcold:
                mode = "http"
            else:
                if is_hot:
                    mode = "hot" if (np.random.rand() < GRAD_HOT_PROB) else "http"
                else:
                    mode = "cold" if (np.random.rand() < GRAD_COLD_PROB) else "http"
            forced_inst = None
            if ABL_CFG.disable_nsga2:
                # [MODIFIED] Baseline Greedy/RoundRobin/Random 也走这里 (inst select)
                # 因为它们都没有 NSGA2
                choice = HYBRID_SCHED.select_instance(func_grad, eid, insts_grad, req_grad)
                if choice and choice[0]:
                    forced_inst = choice[0]
                else:
                    grad_fallback_cnt += 1
            else:
                try:
                    choice = nsga2_select(insts_grad, req_grad, float(CURRENT_DEADLINE_MS),
                                          pop_size=int(os.getenv("NSGA2_POP_SIZE", "30")),
                                          generations=int(os.getenv("NSGA2_GENS", "8")), modes=available_modes)
                except Exception:
                    choice = None
                if choice is None:
                    grad_fallback_cnt += 1;
                    grad_nsga2_feasible = int(grad_nsga2_feasible or 0)
                else:
                    grad_nsga2_feasible = 1;
                    forced_inst, _ = choice
            try:
                inst, (tot, q, cold, net, comp), retry = await invoke_with_retry(
                    func_grad, eid, insts_grad, req_grad, GRAD_BASE_MS, mode=mode, max_tries=INVOKE_RETRIES,
                    forced_inst=forced_inst, global_step=global_step
                )
                grad_mode_counts[mode] += 1;
                grad_inv_total_ms += tot;
                grad_inv_queue_ms += q
                grad_inv_cold_ms += cold;
                grad_inv_net_ms += net;
                grad_inv_compute_ms += comp;
                grad_inv_retry_cnt += int(retry)
                c = _inst_cost_usd(inst, tot);
                cost_usd_grad_apply += c
                grad_lat_sum[mode] += float(tot);
                grad_lat_cnt[mode] += 1;
                grad_bytes_mode[mode] += int(grad_size)
            except RuntimeError:
                grad_mode_counts["fallback"] += 1;
                grad_inv_compute_ms += GRAD_BASE_MS * 2.0

    cold_total = 0;
    cold_skipped = 0;
    cold_updated = 0
    cold_apply_steps_sum = 0.0;
    cold_grad_scale_sum = 0.0;
    cold_pending_steps_sum = 0.0
    cold_pending_cnt = 0;
    cold_update_hit_cnt = 0

    if train:
        OPT_SHARED.step()

        # [MODIFIED] 梯度同步逻辑：支持 BSP, ASP, SSP 和 原有的 Hybrid
        should_sync_global = False

        if ABL_CFG.force_sync_update:
            # BSP (全同步) 或 消融实验的 sync_update
            should_sync_global = True
        elif ABL_CFG.is_asp:
            # ASP (全异步 -> 在模拟中近似为无等待立即更新)
            should_sync_global = True
        elif ABL_CFG.is_ssp:
            # SSP (陈旧度检查)
            max_pending = 0
            for eid in range(MOE_CONFIG.num_experts):
                if _COLD_PENDING[eid] >= SSP_LIMIT:
                    should_sync_global = True  # 只要有一个超标，全局强推 (简化实现)
                    break

        if should_sync_global:
            # 强行更新所有梯度
            OPT_EXPERT.step()
            for eid, expert in enumerate(REAL_MODEL.experts):
                for p in expert.parameters(): p.grad = None
                _COLD_PENDING[eid] = 0
            cold_total = int(MOE_CONFIG.num_experts);
            cold_updated = int(MOE_CONFIG.num_experts);
            cold_update_hit_cnt = int(MOE_CONFIG.num_experts)
        else:
            # 默认：Serverless 异步累积逻辑 (Ours) / 普通的 SSP (未触发阈值时)
            update_eids = set()
            for eid in range(MOE_CONFIG.num_experts):
                is_hot = HEATMAP.is_hot(eid) if HEATMAP else False

                # SSP 的局部触发逻辑
                is_ssp_forced = ABL_CFG.is_ssp and (_COLD_PENDING[eid] >= SSP_LIMIT)

                if is_hot or is_ssp_forced:
                    _COLD_PENDING[eid] = 0;
                    update_eids.add(eid)
                else:
                    cold_total += 1;
                    _COLD_PENDING[eid] += 1
                    cold_pending_steps_sum += float(_COLD_PENDING[eid]);
                    cold_pending_cnt += 1
                    if COLD_ACC_STEPS > 1 and (_COLD_PENDING[eid] % COLD_ACC_STEPS != 0):
                        cold_skipped += 1;
                        continue
                    update_eids.add(eid);
                    cold_update_hit_cnt += 1
            if update_eids:
                for eid in update_eids:
                    if HEATMAP and (not HEATMAP.is_hot(eid)):
                        apply_steps = max(1, _COLD_PENDING[eid]);
                        cold_updated += 1
                        cold_apply_steps_sum += float(apply_steps);
                        cold_grad_scale_sum += float(1.0 / float(apply_steps))
                        if apply_steps > 1:
                            for p in REAL_MODEL.experts[eid].parameters():
                                if p.grad is not None: p.grad.mul_(1.0 / float(apply_steps))
                stashes = {}
                for eid, expert in enumerate(REAL_MODEL.experts):
                    if eid not in update_eids:
                        stash = [];
                        for p in expert.parameters(): stash.append(p.grad); p.grad = None
                        stashes[eid] = stash
                OPT_EXPERT.step()
                for eid, stash in stashes.items():
                    i = 0
                    for p in REAL_MODEL.experts[eid].parameters(): p.grad = stash[i]; i += 1
                for eid in update_eids:
                    for p in REAL_MODEL.experts[eid].parameters(): p.grad = None
                    if HEATMAP and (not HEATMAP.is_hot(eid)): _COLD_PENDING[eid] = 0

    fwd_mode_counts_expert = defaultdict(int);
    fwd_mode_counts_token = defaultdict(int)
    dispatch_fwd = 0;
    capacity_val = 0;
    overflow_total_assign = 0;
    overflow_dropped_assign = 0
    active_hot_set = set();
    active_cold_set = set()
    pre_lat = 0.0;
    post_lat = 0.0;
    exp_comm = 0.0
    inv_total_ms = 0.0;
    inv_queue_ms = 0.0;
    inv_cold_ms = 0.0;
    inv_net_ms = 0.0;
    inv_compute_ms = 0.0;
    inv_retry_cnt = 0
    loss = 0.0;
    acc1 = 0.0;
    acc5 = 0.0
    cost_usd_pre_fwd = 0.0;
    cost_usd_post_fwd = 0.0;
    cost_usd_expert_fwd = 0.0

    for r in results:
        cost_usd_pre_fwd += float(r.get("cost_usd_pre_fwd", 0.0) or 0.0)
        cost_usd_post_fwd += float(r.get("cost_usd_post_fwd", 0.0) or 0.0)
        cost_usd_expert_fwd += float(r.get("cost_usd_expert_fwd", 0.0) or 0.0)
        for mode, count in r["mode_counts_expert"].items(): fwd_mode_counts_expert[mode] += int(count)
        for mode, tok in r["mode_counts_token"].items(): fwd_mode_counts_token[mode] += int(tok)
        dispatch_fwd += int(r["dispatch_count"])
        capacity_val = int(r.get("capacity", capacity_val)) or capacity_val
        overflow_total_assign += int(r.get("overflow_total_assignments", 0))
        overflow_dropped_assign += int(r.get("overflow_dropped_assignments", 0))
        active_hot_set |= set(r["hot_experts"]);
        active_cold_set |= set(r["cold_experts"])
        pre_lat += float(r.get("pre_lat", 0.0));
        post_lat += float(r.get("post_lat", 0.0));
        exp_comm += float(r.get("expert_comm", 0.0))
        inv_total_ms += float(r.get("inv_total_ms", 0.0));
        inv_queue_ms += float(r.get("inv_queue_ms", 0.0))
        inv_cold_ms += float(r.get("inv_cold_ms", 0.0));
        inv_net_ms += float(r.get("inv_net_ms", 0.0))
        inv_compute_ms += float(r.get("inv_compute_ms", 0.0));
        inv_retry_cnt += int(r.get("inv_retry_cnt", 0))
        loss += float(r.get("loss", 0.0));
        acc1 += float(r.get("acc_top1", 0.0));
        acc5 += float(r.get("acc_top5", 0.0))

    n_mb = max(1, len(results));
    loss /= n_mb;
    acc1 /= n_mb;
    acc5 /= n_mb
    safe_dispatch = dispatch_fwd if dispatch_fwd > 0 else 1;
    safe_tok = sum(fwd_mode_counts_token.values()) or 1
    fwd_mode_hot_frac = fwd_mode_counts_expert.get("hot", 0) / safe_dispatch
    fwd_mode_cold_frac = fwd_mode_counts_expert.get("cold", 0) / safe_dispatch
    fwd_mode_local_frac = fwd_mode_counts_expert.get("local", 0) / safe_dispatch
    fwd_mode_hot_frac_tok = fwd_mode_counts_token.get("hot", 0) / safe_tok
    fwd_mode_cold_frac_tok = fwd_mode_counts_token.get("cold", 0) / safe_tok
    fwd_mode_local_frac_tok = fwd_mode_counts_token.get("local", 0) / safe_tok
    active_expert_cnt = len(active_hot_set | active_cold_set)
    active_hot_ratio = (len(active_hot_set) / max(1, active_expert_cnt))
    current_hot_ratio = HEATMAP.hot_ratio() if HEATMAP else 0.0
    hot_flip_cnt = HEATMAP.consume_flip_count() if HEATMAP else 0
    hot_set_size = HEATMAP.hot_set_size() if HEATMAP else 0
    hot_set_jaccard = HEATMAP.hot_set_jaccard() if HEATMAP else 1.0
    expert_load_entropy = HEATMAP.expert_load_entropy() if HEATMAP else 0.0
    overflow_drop_ratio = float(overflow_dropped_assign / max(1, overflow_total_assign))
    samples_per_s = BATCH_SIZE / (step_duration_ms / 1000.0 + 1e-6)
    tokens_per_s = samples_per_s * BLOCK_SIZE
    current_cold_skip_ratio = (cold_skipped / cold_total) if cold_total > 0 else 0.0
    cold_apply_steps_avg = (cold_apply_steps_sum / cold_updated) if cold_updated > 0 else 0.0
    cold_grad_scale_avg = (cold_grad_scale_sum / cold_updated) if cold_updated > 0 else 0.0
    cold_pending_steps_avg = (cold_pending_steps_sum / cold_pending_cnt) if cold_pending_cnt > 0 else 0.0

    grad_mode_hot_frac = grad_mode_counts.get("hot", 0) / max(1, grad_total)
    grad_mode_cold_frac = grad_mode_counts.get("cold", 0) / max(1, grad_total)
    grad_mode_http_frac = grad_mode_counts.get("http", 0) / max(1, grad_total)
    grad_mode_local_frac = grad_mode_counts.get("local", 0) / max(1, grad_total)
    grad_mode_fallback_frac = grad_mode_counts.get("fallback", 0) / max(1, grad_total)

    def _avg_lat(mode: str):
        return float(grad_lat_sum.get(mode, 0.0) / max(1, grad_lat_cnt.get(mode, 0)))

    grad_lat_hot_ms = _avg_lat("hot");
    grad_lat_cold_ms = _avg_lat("cold");
    grad_lat_http_ms = _avg_lat("http")

    deadline_ms = float(CURRENT_DEADLINE_MS if DEADLINE_MODE == "auto" else STEP_PERIOD_MS)
    deadline_slack_ms = float(deadline_ms - step_duration_ms)
    deadline_miss = 1 if deadline_slack_ms < 0 else 0
    cost_usd_step = (
            cost_usd_pre_fwd + cost_usd_post_fwd + cost_usd_expert_fwd + cost_usd_pre_bwd + cost_usd_post_bwd + cost_usd_grad_apply)

    # [MODIFIED] Log Metric: 记录当前是 Baseline 还是 Ablation
    log_mode = f"BASE:{ABL_CFG.baseline_mode}" if ABL_CFG.is_baseline else f"ABL:{ABL_CFG.ablation_mode}"

    metrics_logger.log(
        StepMetrics(
            epoch=epoch, step=global_step, step_in_epoch=step_in_epoch, phase=phase, loss=float(loss),
            acc_top1=float(acc1), acc_top5=float(acc5), batch_size=BATCH_SIZE, seq_len=BLOCK_SIZE,
            tokens=BATCH_SIZE * BLOCK_SIZE,
            step_time_ms=float(step_duration_ms), pre_fwd_ms=float(pre_lat / n_mb), post_fwd_ms=float(post_lat / n_mb),
            expert_comm_ms=float(exp_comm / n_mb), bwd_total_ms=float(bwd_total_ms),
            pre_bwd_ms=float(pre_bwd_ms), post_bwd_ms=float(post_bwd_ms),
            bwd_inv_total_ms=float(bwd_inv_total_ms), bwd_inv_queue_ms=float(bwd_inv_queue_ms),
            bwd_inv_cold_ms=float(bwd_inv_cold_ms), bwd_inv_net_ms=float(bwd_inv_net_ms),
            bwd_inv_compute_ms=float(bwd_inv_compute_ms), bwd_inv_retry_cnt=int(bwd_inv_retry_cnt),
            samples_per_s=float(samples_per_s), tokens_per_s=float(tokens_per_s), grad_bytes=float(grad_bytes),
            grad_total=int(grad_total), grad_mode_hot_frac=float(grad_mode_hot_frac),
            grad_mode_cold_frac=float(grad_mode_cold_frac),
            grad_mode_http_frac=float(grad_mode_http_frac), grad_mode_local_frac=float(grad_mode_local_frac),
            grad_mode_fallback_frac=float(grad_mode_fallback_frac), grad_inv_total_ms=float(grad_inv_total_ms),
            grad_inv_queue_ms=float(grad_inv_queue_ms), grad_inv_cold_ms=float(grad_inv_cold_ms),
            grad_inv_net_ms=float(grad_inv_net_ms), grad_inv_compute_ms=float(grad_inv_compute_ms),
            grad_inv_retry_cnt=int(grad_inv_retry_cnt), grad_nsga2_feasible=int(grad_nsga2_feasible),
            grad_fallback_cnt=int(grad_fallback_cnt), grad_lat_hot_ms=float(grad_lat_hot_ms),
            grad_lat_cold_ms=float(grad_lat_cold_ms), grad_lat_http_ms=float(grad_lat_http_ms),
            grad_bytes_hot=int(grad_bytes_mode.get("hot", 0)), grad_bytes_cold=int(grad_bytes_mode.get("cold", 0)),
            grad_bytes_http=int(grad_bytes_mode.get("http", 0)), dispatch_count=int(dispatch_fwd),
            expert_inst_cnt=int(MOE_CONFIG.num_experts), hot_ratio=float(current_hot_ratio),
            active_expert_cnt=int(active_expert_cnt), active_hot_ratio=float(active_hot_ratio),
            hot_flip_cnt=int(hot_flip_cnt), hot_set_size=int(hot_set_size), hot_set_jaccard=float(hot_set_jaccard),
            expert_load_entropy=float(expert_load_entropy), cold_total_cnt=int(cold_total),
            cold_skipped_cnt=int(cold_skipped), cold_updated_cnt=int(cold_updated),
            cold_skip_ratio=float(current_cold_skip_ratio), cold_apply_steps_avg=float(cold_apply_steps_avg),
            cold_grad_scale_avg=float(cold_grad_scale_avg), cold_pending_steps_avg=float(cold_pending_steps_avg),
            cold_update_hit_cnt=int(cold_update_hit_cnt), fwd_mode_hot_frac=float(fwd_mode_hot_frac),
            fwd_mode_cold_frac=float(fwd_mode_cold_frac), fwd_mode_local_frac=float(fwd_mode_local_frac),
            fwd_mode_hot_frac_tok=float(fwd_mode_hot_frac_tok), fwd_mode_cold_frac_tok=float(fwd_mode_cold_frac_tok),
            fwd_mode_local_frac_tok=float(fwd_mode_local_frac_tok), capacity=int(capacity_val),
            overflow_total_assignments=int(overflow_total_assign),
            overflow_dropped_assignments=int(overflow_dropped_assign), overflow_drop_ratio=float(overflow_drop_ratio),
            inv_total_ms=float(inv_total_ms), inv_queue_ms=float(inv_queue_ms), inv_cold_ms=float(inv_cold_ms),
            inv_net_ms=float(inv_net_ms), inv_compute_ms=float(inv_compute_ms), inv_retry_cnt=int(inv_retry_cnt),
            deadline_ms=float(deadline_ms), deadline_miss=int(deadline_miss),
            deadline_slack_ms=float(deadline_slack_ms),
            cost_usd_pre_fwd=float(cost_usd_pre_fwd), cost_usd_post_fwd=float(cost_usd_post_fwd),
            cost_usd_expert_fwd=float(cost_usd_expert_fwd), cost_usd_pre_bwd=float(cost_usd_pre_bwd),
            cost_usd_post_bwd=float(cost_usd_post_bwd), cost_usd_grad_apply=float(cost_usd_grad_apply),
            cost_usd_step=float(cost_usd_step), ablation_mode=log_mode,
            grad_selector="nsga2" if (USE_NSGA2 and not ABL_CFG.disable_nsga2) else "heuristic",
        )
    )

    global _metric_buffer, _metric_count
    if train:
        _metric_buffer["loss"] += float(loss)
        _metric_buffer["step_time"] += float(step_duration_ms)
        _metric_buffer["cost"] += float(cost_usd_step)
        _metric_count += 1

        if _metric_count >= LOG_TRAIN_EVERY:
            avg_loss = _metric_buffer["loss"] / _metric_count
            avg_time = _metric_buffer["step_time"] / _metric_count
            avg_cost = _metric_buffer["cost"] / _metric_count
            print(
                f"[Step {global_step}/{MAX_STEPS}] "
                f"Loss(avg{LOG_TRAIN_EVERY}): {avg_loss:.4f} | "
                f"Time(avg{LOG_TRAIN_EVERY}): {avg_time:.0f}ms | "
                f"Deadline(now): {CURRENT_DEADLINE_MS:.0f}ms | "
                f"Cost(avg{LOG_TRAIN_EVERY}): {avg_cost:.8f} | "
                f"SLO(miss/slack): {deadline_miss}/{deadline_slack_ms:.1f}ms"
            )
            _metric_buffer = defaultdict(float)
            _metric_count = 0


# ============================================================
# Main
# ============================================================
async def main():
    log("controller", "Starting controller (Unified Ablation + Baseline Mode) ...")

    global REAL_MODEL, OPT_SHARED, OPT_EXPERT, HEATMAP
    if not SimpleMoE:
        raise RuntimeError("moe_model.SimpleMoE not found")

    REAL_MODEL = SimpleMoE(
        vocab_size=VOCAB_SIZE, d_model=MOE_CONFIG.d_model,
        num_experts=MOE_CONFIG.num_experts, top_k=MOE_CONFIG.top_k,
    )

    shared_params = []
    if hasattr(REAL_MODEL, "pre_stage"): shared_params.extend(list(REAL_MODEL.pre_stage.parameters()))
    if hasattr(REAL_MODEL, "post_stage"): shared_params.extend(list(REAL_MODEL.post_stage.parameters()))
    for name in ["embed", "gate", "norm", "head", "lm_head"]:
        m = getattr(REAL_MODEL, name, None)
        if m is not None: shared_params.extend(list(m.parameters()))

    expert_params = []
    for expert in REAL_MODEL.experts: expert_params += list(expert.parameters())

    OPT_SHARED = optim.Adam(shared_params, lr=LR)
    OPT_EXPERT = optim.Adam(expert_params, lr=LR)

    alpha_short = float(os.getenv("ALPHA_SHORT", "0.45"))
    alpha_long = float(os.getenv("ALPHA_LONG", "0.05"))
    high_mul = float(os.getenv("HOT_HIGH_MUL", "1.25"))
    low_mul = float(os.getenv("HOT_LOW_MUL", "0.80"))
    trend_discount = float(os.getenv("TREND_DISCOUNT", "0.80"))

    HEATMAP = AdaptiveHysteresisHeatmap(
        MOE_CONFIG.num_experts, alpha_short=alpha_short, alpha_long=alpha_long,
        high_mul=high_mul, low_mul=low_mul, trend_discount=trend_discount,
    )
    if ABL_CFG.disable_hotcold: HEATMAP = None

    train_batcher = LMTextBatcher(data_path=DATA_PATH, split="train", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
    val_batcher = LMTextBatcher(data_path=DATA_PATH, split="val", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
    total_data_size = len(train_batcher.data) if hasattr(train_batcher, "data") else 1
    steps_per_epoch = max(1, int(total_data_size) // (BATCH_SIZE * BLOCK_SIZE))
    metrics_logger = MetricsLogger("metrics.csv", tail_window=int(os.getenv("TAIL_WINDOW", "50")))

    global_step = 0
    while global_step < MAX_STEPS:
        if global_step > 0 and global_step % 200 == 0:
            log("controller",
                f">>> [System Idle] Simulating traffic gap at step {global_step}. All instances cooling down...")
            async with INSTANCE_LOCK:
                INSTANCE_MGR.last_step_access.clear()

        await run_step("train", train_batcher, global_step, metrics_logger, steps_per_epoch)
        global_step += 1
        if global_step % VAL_INTERVAL == 0:
            await run_step("val", val_batcher, global_step, metrics_logger, steps_per_epoch)

    log("controller", "Training Finished.")


if __name__ == "__main__":
    asyncio.run(main())