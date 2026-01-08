# -------------------------------------------------------------------------
# [UPDATED - FULL REPLACE controller.py]
# 基于你当前 controller.py（已读入）做的增强：
# 1) 保留你现有的 Ablation/Baseline + 冷启动/网络/队列仿真 breakdown + HTTP 执行闭环
# 2) 新增 TriScheduler：Heuristic + OnlinePred + NSGA(Pareto) 三段调度链路
# 3) 新增消融：no_nsga / no_online / no_heuristic
# 4) invoke_with_retry 统一改为 TRI_SCHED.select(...)，保证 pre/post/expert 全链路消融生效
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import asyncio
import json
import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ✅ 使用你项目既有的序列化协议
from shared import dumps, loads, tensor_to_pack, pack_to_tensor

# ============================================================
# Global Env
# ============================================================

SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

COMM_SIM_DIR = os.getenv("COMM_SIM_DIR", "comm_sim")
INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")

EXPERIMENT_TYPE = os.getenv("EXPERIMENT_TYPE", "ablation")  # ablation / baseline
ABLATION_MODE = os.getenv("ABLATION_MODE", "full")
BASELINE_MODE = os.getenv("BASELINE_MODE", "round_robin")

# Training
MAX_STEPS = int(os.getenv("MAX_STEPS", "800"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MICRO_BATCH = int(os.getenv("MICRO_BATCH", "4"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "20"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))

# Sequence / token LM batch（真实训练用）
SEQ_LEN = int(os.getenv("SEQ_LEN", "64"))
DATA_PATH = os.getenv("DATA_PATH", "input.txt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")

# MoE
NUM_EXPERTS = int(os.getenv("NUM_EXPERTS", "4"))
TOP_K = int(os.getenv("TOP_K", "2"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "2000"))  # fallback/校验
EMB_DIM = int(os.getenv("EMB_DIM", "256"))

# Hot/Cold logic
HOTSET_SIZE = int(os.getenv("HOTSET_SIZE", "1"))
HEATMAP_DECAY = float(os.getenv("HEATMAP_DECAY", "0.98"))
HEATMAP_MIN_PROB = float(os.getenv("HEATMAP_MIN_PROB", "0.01"))

# Traffic skew (hotspot drift)
HOTSPOT_DRIFT_EVERY = int(os.getenv("HOTSPOT_DRIFT_EVERY", "50"))
HOTSPOT_SPAN = int(os.getenv("HOTSPOT_SPAN", "1"))
HOT_PROB = float(os.getenv("HOT_PROB", "0.85"))
WARM_PROB = float(os.getenv("WARM_PROB", "0.15"))
TRAFFIC_SKEW_ENABLE = os.getenv("TRAFFIC_SKEW_ENABLE", "1") == "1"

# Network multipliers
DEFAULT_NET_LATENCY = float(os.getenv("DEFAULT_NET_LATENCY_MS", "5.0"))
DEFAULT_PERFORMANCE = float(os.getenv("DEFAULT_PERFORMANCE", "1.0"))
HOT_NET_MUL = float(os.getenv("HOT_NET_MUL", "0.5"))
COLD_NET_MUL = float(os.getenv("COLD_NET_MUL", "2.0"))
HTTP_NET_MUL = float(os.getenv("HTTP_NET_MUL", "1.0"))
FALLBACK_NET_MUL = float(os.getenv("FALLBACK_NET_MUL", "1.3"))
COLD_STORAGE_MS = float(os.getenv("COLD_STORAGE_MS", "12.0"))

# Retry & SLO
INVOKE_RETRIES = int(os.getenv("INVOKE_RETRIES", "10"))
DEADLINE_WARMUP_STEPS = int(os.getenv("DEADLINE_WARMUP_STEPS", "30"))
DEADLINE_PCTL = int(os.getenv("DEADLINE_PCTL", "95"))
DEADLINE_SAFETY = float(os.getenv("DEADLINE_SAFETY", "1.10"))
DEADLINE_MIN_MS = float(os.getenv("DEADLINE_MIN_MS", "200"))

# Baseline SSP limit (placeholder)
SSP_LIMIT = int(os.getenv("SSP_LIMIT", "4"))

# Autoscale
AUTOSCALE_ENABLE = os.getenv("AUTOSCALE_ENABLE", "1") == "1"
AUTOSCALE_QUEUE_TH_MS = float(os.getenv("AUTOSCALE_QUEUE_TH_MS", "30"))
AUTOSCALE_MAX_REPLICA = int(os.getenv("AUTOSCALE_MAX_REPLICA", "6"))
AUTOSCALE_COOLDOWN_STEPS = int(os.getenv("AUTOSCALE_COOLDOWN_STEPS", "8"))

# ---- HTTP serverless execution ----
USE_HTTP_EXEC = os.getenv("USE_HTTP_EXEC", "1") == "1"
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "30"))
SIM_SLEEP = os.getenv("SIM_SLEEP", "0") == "1"
HTTP_CONCURRENCY = int(os.getenv("HTTP_CONCURRENCY", "32"))

# endpoint paths
PATH_FWD = os.getenv("PATH_FWD", "/fwd")
PATH_BWD = os.getenv("PATH_BWD", "/bwd")
PATH_STEP = os.getenv("PATH_STEP", "/step")
PATH_ZERO = os.getenv("PATH_ZERO", "/zero")
PATH_HEALTH = os.getenv("PATH_HEALTH", "/health")

# Async backward policy（热立即 step，冷累计 step）
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "4"))
FORCE_SYNC_UPDATE = os.getenv("FORCE_SYNC_UPDATE", "0") == "1"

# Optional device (for local tensor ops in controller side)
DEVICE = os.getenv("DEVICE", "cpu")

# ============================================================
# Ablation config
# ============================================================

@dataclass
class AblationConfig:
    is_baseline: bool = False
    is_static_compute: bool = False
    disable_hotcold: bool = False
    force_sync_update: bool = False
    use_random_sched: bool = False
    use_rr_sched: bool = False
    use_greedy_sched: bool = False
    use_bsp: bool = False
    use_ssp: bool = False
    use_asp: bool = False

    # ===== NEW: scheduler-chain ablations =====
    disable_nsga: bool = False          # 去 NSGA/Pareto 多目标筛选
    disable_online_pred: bool = False   # 去在线预测（EMA / online stats）
    disable_heuristic: bool = False     # 去启发式估计（cold/net/compute/cost 的静态估计）

    @staticmethod
    def from_env() -> "AblationConfig":
        cfg = AblationConfig()
        if EXPERIMENT_TYPE == "baseline":
            cfg.is_baseline = True
            m = BASELINE_MODE.lower()
            cfg.use_rr_sched = (m == "round_robin")
            cfg.use_random_sched = (m == "random")
            cfg.is_static_compute = (m == "static")
            cfg.use_greedy_sched = (m == "greedy")
            cfg.use_bsp = (m == "bsp")
            cfg.use_ssp = (m == "ssp")
            cfg.use_asp = (m == "asp")
        else:
            m = ABLATION_MODE.lower()

            # ---- existing ablations ----
            if m == "no_hotcold":
                cfg.disable_hotcold = True
            elif m == "sync_update":
                cfg.force_sync_update = True
            elif m == "static_compute":
                cfg.is_static_compute = True
            elif m == "random_sched":
                cfg.use_random_sched = True
            elif m == "rr_sched":
                cfg.use_rr_sched = True
            elif m == "greedy_sched":
                cfg.use_greedy_sched = True

            # ---- NEW ablations ----
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
# Load instances / func map
# ============================================================

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


INSTANCES = _load_json(INSTANCES_FILE)
FUNC_MAP = _load_json(FUNC_MAP_FILE)
INST_BY_ID: Dict[str, Dict[str, Any]] = {x["id"]: x for x in INSTANCES}


def _inst_url(inst: Dict[str, Any]) -> str:
    if "url" in inst and inst["url"]:
        return str(inst["url"]).rstrip("/")
    meta = inst.get("meta", {}) or {}
    host = meta.get("host", inst.get("host", "127.0.0.1"))
    port = meta.get("port", inst.get("port", None))
    if port is None:
        ep = meta.get("endpoint", None)
        if ep:
            s = str(ep)
            if s.startswith("http"):
                return s.rstrip("/")
            return f"http://{s}".rstrip("/")
        raise RuntimeError(f"Instance missing url/port: {inst}")
    return f"http://{host}:{int(port)}".rstrip("/")


# ============================================================
# Heatmap for hot/cold experts
# ============================================================

class HotColdHeatmap:
    def __init__(self, num_experts: int, decay: float = 0.98, min_prob: float = 0.01):
        self.num_experts = num_experts
        self.decay = decay
        self.min_prob = min_prob
        self.scores = np.ones(num_experts, dtype=np.float32) / num_experts

    def update_from_routing(self, topk_idx: torch.Tensor, topk_vals: torch.Tensor):
        idx = topk_idx.reshape(-1).detach().cpu().numpy()
        vals = topk_vals.reshape(-1).detach().cpu().numpy()
        add = np.zeros(self.num_experts, dtype=np.float32)
        for e, v in zip(idx, vals):
            add[int(e)] += float(v)

        self.scores *= self.decay
        self.scores += (1.0 - self.decay) * add
        self.scores = np.maximum(self.scores, self.min_prob)
        self.scores /= (self.scores.sum() + 1e-9)

    def hot_set(self, k: int) -> List[int]:
        k = max(1, min(k, self.num_experts))
        return list(np.argsort(-self.scores)[:k])


HEATMAP = HotColdHeatmap(NUM_EXPERTS, HEATMAP_DECAY, HEATMAP_MIN_PROB)
HEATMAP_LOCK = asyncio.Lock()
PREV_HOT_SET = None


# ============================================================
# Utils
# ============================================================

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
# Cold/Warm simulation (REAL-TIME idle based)
# ============================================================

class InstanceManager:
    def __init__(self):
        self.last_used_ts_ms: Dict[str, float] = {}
        self.lock = asyncio.Lock()

        self.keep_alive_ms = float(os.getenv("KEEP_ALIVE_MS", "5000"))
        self.eviction_base_prob = float(os.getenv("EVICTION_BASE_PROB", "0.02"))
        self.eviction_tau_ms = float(os.getenv("EVICTION_TAU_MS", "20000"))

        self.keepalive_mul_hot = float(os.getenv("KEEPALIVE_MUL_HOT", "1.5"))
        self.keepalive_mul_cold = float(os.getenv("KEEPALIVE_MUL_COLD", "0.7"))
        self.keepalive_mul_http = float(os.getenv("KEEPALIVE_MUL_HTTP", "1.0"))

    def _keepalive_mul(self, mode: str) -> float:
        m = (mode or "").lower()
        if m == "hot":
            return self.keepalive_mul_hot
        if m == "cold":
            return self.keepalive_mul_cold
        return self.keepalive_mul_http

    def _default_cold_start_ms(self, func_name: str, inst: Dict[str, Any]) -> float:
        fn = (func_name or "").lower()
        region = str(inst.get("region", "local")).lower()
        is_local = ("local" in region)
        if "expert" in fn:
            return 80.0 if is_local else 450.0
        if "pre_" in fn or "post_" in fn or "pre" in fn or "post" in fn:
            return 40.0 if is_local else 250.0
        if "apply_grad" in fn or "grad" in fn:
            return 60.0 if is_local else 300.0
        return 40.0 if is_local else 250.0

    async def cold_start_ms(self, inst: Dict[str, Any], *, func_name: str, mode: str) -> float:
        if ABL_CFG.is_static_compute:
            return 0.0

        inst_id = str(inst.get("id", ""))
        meta = inst.get("meta", {}) or {}
        raw = meta.get("cold_start_ms", None)
        if raw is None:
            cold_ms = self._default_cold_start_ms(func_name, inst)
        else:
            try:
                cold_ms = float(raw)
            except Exception:
                cold_ms = self._default_cold_start_ms(func_name, inst)
            if cold_ms <= 0:
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
                if random.random() < p:
                    is_cold = True

            self.last_used_ts_ms[inst_id] = now_ms

        return cold_ms if is_cold else 0.0


INSTANCE_MGR = InstanceManager()

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
    if cpu is not None:
        try:
            return max(1, int(cpu))
        except Exception:
            return 1
    return INSTANCE_MAX_CONC_DEFAULT


def _get_inst_sem(inst: Dict[str, Any]) -> asyncio.Semaphore:
    inst_id = inst.get("id")
    if inst_id not in INSTANCE_SEM:
        INSTANCE_SEM[inst_id] = asyncio.Semaphore(_get_inst_max_conc(inst))
    return INSTANCE_SEM[inst_id]


# ============================================================
# Invoke Simulation (breakdown) + HTTP execution
# ============================================================

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
    返回：(total_ms, queue_ms, cold_ms, net_ms, compute_ms)
    """
    region = str(inst.get("region", "local")).lower()
    is_local = "local" in region

    local_oom_prob = float(os.getenv("LOCAL_OOM_PROB", "0.05"))
    if ABL_CFG.is_static_compute and is_local:
        local_oom_prob = 0.0

    if is_local and local_oom_prob > 0 and random.random() < local_oom_prob:
        raise RuntimeError(f"Simulated Local Busy/OOM for {inst.get('id')}")

    meta = inst.get("meta", {}) or {}
    perf = float(meta.get("performance", DEFAULT_PERFORMANCE))
    raw_compute = float(base_compute_ms) / max(perf, 1e-6)
    compute_ms = raw_compute * random.uniform(0.90, 1.10)

    cold_ms = float(await INSTANCE_MGR.cold_start_ms(inst, func_name=func_name, mode=mode))

    sem = _get_inst_sem(inst)
    tq0 = time.perf_counter()
    async with sem:
        queue_ms = (time.perf_counter() - tq0) * 1000.0

        net_base = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))
        net_ms = net_base * _mode_net_multiplier(mode) * random.uniform(0.90, 1.10)
        if (mode or "").lower() == "cold":
            net_ms += float(COLD_STORAGE_MS)

        total_ms = queue_ms + cold_ms + net_ms + compute_ms
        return total_ms, queue_ms, cold_ms, net_ms, compute_ms


# ----------------- HTTP wrapper -----------------
try:
    import aiohttp  # type: ignore
except Exception:
    aiohttp = None  # type: ignore

import requests

_HTTP_SEM = asyncio.Semaphore(max(1, HTTP_CONCURRENCY))


async def invoke_http(
    inst: Dict[str, Any],
    *,
    path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Request JSON:
      {"trace_id": "...", "payload": <packed-dict>}

    Response JSON:
      {"trace_id": "...", "payload": <packed-dict>}  或者扁平 dict（兼容旧版本）
    """
    url = _inst_url(inst).rstrip("/")
    full = url + path

    trace_id = None
    if isinstance(payload, dict):
        trace_id = payload.get("trace_id") or payload.get("trace") or payload.get("_trace_id")
    if not trace_id:
        trace_id = f"tr_{int(time.time()*1000)}_{random.randint(0, 10**9)}"

    def _pack_any(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return tensor_to_pack(x.detach().cpu())
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, dict):
            return {str(k): _pack_any(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_pack_any(v) for v in x]
        return x

    def _unpack_any(x: Any) -> Any:
        if isinstance(x, dict):
            if ("shape" in x) and (("dtype" in x) or ("t" in x)) and ("data" in x):
                try:
                    return pack_to_tensor(x).to("cpu")
                except Exception:
                    pass
            return {k: _unpack_any(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_unpack_any(v) for v in x]
        return x

    pure_payload = dict(payload) if isinstance(payload, dict) else {"_raw": payload}
    pure_payload["trace_id"] = trace_id
    body_obj = {"trace_id": trace_id, "payload": _pack_any(pure_payload)}

    async with _HTTP_SEM:
        def _do():
            r = requests.post(full, json=body_obj, timeout=HTTP_TIMEOUT_S)

            if r.status_code >= 400:
                print("\n================ HTTP ERROR ================\n")
                print("URL:", full)
                print("STATUS:", r.status_code)
                print("RESPONSE TEXT:", (r.text or "")[:4000])
                print("\n=============== REQUEST BODY ===============\n")
                try:
                    print(json.dumps(body_obj, ensure_ascii=False)[:4000])
                except Exception:
                    print(str(body_obj)[:4000])
                print("\n===========================================\n")

            r.raise_for_status()
            obj = r.json() if r.content else {}
            if not isinstance(obj, dict):
                return {"ok": True, "trace_id": trace_id, "_raw": obj}

            if "payload" in obj and isinstance(obj["payload"], dict):
                inner = _unpack_any(obj["payload"])
                if not isinstance(inner, dict):
                    inner = {"_payload": inner}
                inner.setdefault("trace_id", obj.get("trace_id", trace_id))
                inner.setdefault("ok", obj.get("ok", True))
                return inner

            obj.setdefault("trace_id", obj.get("trace_id", trace_id))
            return _unpack_any(obj)

        return await asyncio.to_thread(_do)


# ============================================================
# Baseline schedulers
# ============================================================

class BaselineScheduler:
    def __init__(self):
        self.rr_ptr: Dict[str, int] = {}

    def select_random(self, inst_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return random.choice(inst_list)

    def select_rr(self, func_name: str, inst_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not inst_list:
            raise RuntimeError("empty inst_list")
        p = self.rr_ptr.get(func_name, 0) % len(inst_list)
        self.rr_ptr[func_name] = p + 1
        return inst_list[p]

    def select_greedy_min_rtt(self, inst_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = None
        best_v = 1e18
        for x in inst_list:
            meta = x.get("meta", {}) or {}
            rtt = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))
            if rtt < best_v:
                best_v = rtt
                best = x
        return best if best is not None else random.choice(inst_list)


BASELINE_SCHED = BaselineScheduler()


# ============================================================
# Our hybrid scheduler (online stats / EMA)
# ============================================================

class HybridScheduler:
    def __init__(self):
        self.ema_lat: Dict[str, float] = {}
        self.ema_decay = float(os.getenv("SCHED_EMA_DECAY", "0.95"))

    def update_stats(self, inst: Dict[str, Any], tot_ms: float):
        inst_id = inst.get("id")
        v = self.ema_lat.get(inst_id)
        if v is None:
            self.ema_lat[inst_id] = float(tot_ms)
        else:
            self.ema_lat[inst_id] = self.ema_decay * v + (1.0 - self.ema_decay) * float(tot_ms)

    def select_best(self, inst_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = None
        best_v = 1e18
        for x in inst_list:
            inst_id = x.get("id")
            v = self.ema_lat.get(inst_id)
            if v is None:
                meta = x.get("meta", {}) or {}
                v = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY))) * 10.0
            if v < best_v:
                best_v = v
                best = x
        return best if best is not None else random.choice(inst_list)


HYBRID_SCHED = HybridScheduler()


# ============================================================
# Deadline estimator
# ============================================================

class DeadlineEstimator:
    def __init__(self):
        self.hist: List[float] = []

    def update(self, step_ms: float):
        self.hist.append(float(step_ms))
        if len(self.hist) > 200:
            self.hist.pop(0)

    def deadline_ms(self, step: int) -> float:
        if step < DEADLINE_WARMUP_STEPS or len(self.hist) < 10:
            return float(DEADLINE_MIN_MS)
        p = np.percentile(self.hist, DEADLINE_PCTL)
        return float(max(DEADLINE_MIN_MS, p * DEADLINE_SAFETY))

DEADLINE_EST = DeadlineEstimator()


# ============================================================
# Cost model
# ============================================================

def _cost_usd(inst: Dict[str, Any], dur_ms: float) -> float:
    meta = inst.get("meta", {}) or {}
    cents_s = float(meta.get("price_cents_s", 0.0))
    return (cents_s / 100.0) * (dur_ms / 1000.0)


# ============================================================
# TriScheduler = Heuristic + OnlinePred + (NSGA-like Pareto)
# ============================================================

SCHED_W_LAT = float(os.getenv("SCHED_W_LAT", "1.0"))
SCHED_W_COST = float(os.getenv("SCHED_W_COST", "0.15"))
SCHED_W_COLD = float(os.getenv("SCHED_W_COLD", "0.25"))
SCHED_W_QUEUE = float(os.getenv("SCHED_W_QUEUE", "0.05"))
NSGA_SEED = int(os.getenv("NSGA_SEED", "42"))

def _predict_static_total_ms_and_cost(
    inst: Dict[str, Any],
    *,
    func_name: str,
    mode: str,
    base_compute_ms: float,
) -> Tuple[float, float, float, float, float]:
    """
    启发式静态估计：返回 (tot_ms, cost_usd, queue_ms, cold_ms, net_ms)
    compute_ms 也折进 tot_ms 里
    """
    meta = inst.get("meta", {}) or {}
    perf = float(meta.get("performance", DEFAULT_PERFORMANCE))
    compute_ms = float(base_compute_ms) / max(perf, 1e-6)

    net_base = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))
    net_ms = net_base * _mode_net_multiplier(mode)
    if (mode or "").lower() == "cold":
        net_ms += float(COLD_STORAGE_MS)

    cold_ms = INSTANCE_MGR._default_cold_start_ms(func_name, inst) if not ABL_CFG.is_static_compute else 0.0
    queue_ms = 0.0

    tot_ms = float(queue_ms + cold_ms + net_ms + compute_ms)
    cost = _cost_usd(inst, tot_ms)
    return tot_ms, cost, queue_ms, cold_ms, net_ms


class TriScheduler:
    """
    三段链路：
      - heuristic: 静态估计 lat/cost/cold
      - online_pred: HYBRID_SCHED.ema_lat 替换 lat（可消融 no_online）
      - nsga: Pareto front 多目标筛选（可消融 no_nsga）
    """
    def __init__(self):
        self.rng = random.Random(NSGA_SEED)

    def _online_lat(self, inst: Dict[str, Any]) -> Optional[float]:
        if ABL_CFG.disable_online_pred:
            return None
        return HYBRID_SCHED.ema_lat.get(inst.get("id"))

    def _heuristic(self, inst: Dict[str, Any], *, func_name: str, mode: str, base_compute_ms: float) -> Tuple[float, float, float, float]:
        # (lat_ms, cost_usd, cold_ms, queue_ms)
        if ABL_CFG.disable_heuristic:
            return 1e9, 0.0, 0.0, 0.0
        tot_ms, cost, queue_ms, cold_ms, _net = _predict_static_total_ms_and_cost(
            inst, func_name=func_name, mode=mode, base_compute_ms=base_compute_ms
        )
        return float(tot_ms), float(cost), float(cold_ms), float(queue_ms)

    @staticmethod
    def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        # minimize (lat, cost)
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _pareto_front(self, pts: List[Tuple[float, float]]) -> List[int]:
        front = []
        for i, p in enumerate(pts):
            dominated = False
            for j, q in enumerate(pts):
                if j == i:
                    continue
                if self._dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        return front

    def _score(self, lat_ms: float, cost_usd: float, cold_ms: float, queue_ms: float) -> float:
        return (
            SCHED_W_LAT * float(lat_ms)
            + SCHED_W_COST * float(cost_usd) * 1000.0
            + SCHED_W_COLD * float(cold_ms)
            + SCHED_W_QUEUE * float(queue_ms)
        )

    def select(
        self,
        inst_list: List[Dict[str, Any]],
        *,
        func_name: str,
        mode: str,
        base_compute_ms: float,
        deadline_ms: float,
    ) -> Dict[str, Any]:
        if not inst_list:
            raise RuntimeError("empty inst_list")

        feats: List[Tuple[float, float, float, float]] = []
        for inst in inst_list:
            h_lat, h_cost, h_cold, h_queue = self._heuristic(inst, func_name=func_name, mode=mode, base_compute_ms=base_compute_ms)
            o_lat = self._online_lat(inst)
            lat = float(o_lat) if (o_lat is not None) else float(h_lat)

            # 双禁用：online+heuristic 都关 → 用 rtt 做一个退化估计
            if (o_lat is None) and ABL_CFG.disable_heuristic:
                meta = inst.get("meta", {}) or {}
                lat = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY))) * 10.0

            feats.append((lat, h_cost, h_cold, h_queue))

        ok = [i for i, (lat, _c, _cold, _q) in enumerate(feats) if lat <= float(deadline_ms)]
        cand_idx = ok if ok else list(range(len(inst_list)))

        # no_nsga：直接按 score 选最小
        if ABL_CFG.disable_nsga or len(cand_idx) <= 1:
            best_i = None
            best_s = 1e18
            for i in cand_idx:
                lat, cost, cold, q = feats[i]
                s = self._score(lat, cost, cold, q)
                if s < best_s:
                    best_s = s
                    best_i = i
            return inst_list[int(best_i)]

        pts = [(feats[i][0], feats[i][1]) for i in cand_idx]  # (lat, cost)
        front_local = self._pareto_front(pts)
        front = [cand_idx[j] for j in front_local]

        best_i = None
        best_s = 1e18
        for i in front:
            lat, cost, cold, q = feats[i]
            s = self._score(lat, cost, cold, q)
            if s < best_s:
                best_s = s
                best_i = i
        return inst_list[int(best_i)]


TRI_SCHED = TriScheduler()


# ============================================================
# Invocation wrapper with retry + baseline interception
# ============================================================

def _maybe_autoscale(func_name: str, candidates: list, queue_ms: float, global_step: int) -> None:
    return


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
    if not candidates:
        raise RuntimeError(f"invoke candidates empty for {func_name}")

    if forced_inst is None and ABL_CFG.is_baseline:
        if ABL_CFG.use_random_sched:
            forced_inst = BASELINE_SCHED.select_random(candidates)
        elif ABL_CFG.use_rr_sched:
            forced_inst = BASELINE_SCHED.select_rr(func_name, candidates)
        elif ABL_CFG.use_greedy_sched:
            forced_inst = BASELINE_SCHED.select_greedy_min_rtt(candidates)
        elif ABL_CFG.is_static_compute:
            forced_inst = candidates[0]

    tries = 0
    last_err: Optional[Exception] = None

    async def _try(inst: Dict[str, Any], retry_cnt: int):
        breakdown = await simulate_invoke_with_breakdown(
            inst, base_compute_ms, req, func_name=func_name, mode=mode, global_step=global_step
        )
        HYBRID_SCHED.update_stats(inst, breakdown[0])
        _maybe_autoscale(func_name, candidates, breakdown[1], global_step)

        if SIM_SLEEP:
            await asyncio.sleep(breakdown[0] / 1000.0)
        return inst, breakdown, retry_cnt

    if forced_inst is not None:
        try:
            return await _try(forced_inst, 0)
        except Exception as e:
            last_err = e

    cand = list(candidates)
    while tries < max_tries and cand:
        tries += 1

        if not ABL_CFG.is_baseline:
            deadline_ms = float(DEADLINE_EST.deadline_ms(global_step))
            inst = TRI_SCHED.select(
                cand,
                func_name=func_name,
                mode=mode,
                base_compute_ms=base_compute_ms,
                deadline_ms=deadline_ms,
            )
        else:
            inst = random.choice(cand)

        try:
            return await _try(inst, max(0, tries - 1))
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

    phase = max(0, int(global_step // max(1, HOTSPOT_DRIFT_EVERY)))
    hot0 = (phase * max(1, HOTSPOT_SPAN)) % num_experts
    hot_set = [(hot0 + i) % num_experts for i in range(max(1, HOTSPOT_SPAN))]
    warm_e = (hot0 + max(1, HOTSPOT_SPAN)) % num_experts

    new_idx = topk_idx_3d.clone()
    new_vals = topk_vals_3d.clone()

    rand_vals = torch.rand((B, T), device=device)
    mask_hot = (rand_vals < HOT_PROB)
    mask_warm = (rand_vals >= HOT_PROB) & (rand_vals < (HOT_PROB + WARM_PROB))
    mask_others = ~(mask_hot | mask_warm)

    if K >= 1:
        t0 = new_idx[..., 0]
        if mask_hot.any():
            t0[mask_hot] = random.choice(hot_set)
        if mask_warm.any():
            t0[mask_warm] = warm_e
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
        if mask_hot.any():
            t1[mask_hot] = target_e
        if mask_warm.any():
            t1[mask_warm] = hot_set[0]
        if mask_others.any():
            t1[mask_others] = warm_e
        new_idx[..., 1] = t1

        v1 = new_vals[..., 1]
        v1[mask_hot] = 0.9
        v1[mask_warm] = 0.3
        v1[mask_others] = 0.4
        new_vals[..., 1] = v1

    if is_2d:
        return new_idx.squeeze(1), new_vals.squeeze(1)
    return new_idx, new_vals


# ============================================================
# Metrics
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

METRICS_FILE = os.getenv("METRICS_FILE", "metrics.csv")

def write_metrics_header(path: str):
    _ensure_dir(path)
    cols = [
        "step", "split",
        "loss", "acc_top1", "acc_top5",
        "step_time_ms",
        "pre_lat_ms", "post_lat_ms", "exp_lat_ms",
        "inv_total_ms", "inv_queue_ms", "inv_cold_ms", "inv_net_ms", "inv_compute_ms",
        "inv_retry_cnt",
        "hot_ratio", "hot_set_changed", "hot_set_jaccard", "fwd_mode_hot_frac", "fwd_mode_cold_frac", "fwd_mode_http_frac",
        "grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac",
        "deadline_ms", "deadline_violation_frac",
        "cost_usd_step", "cost_usd_pre_fwd", "cost_usd_post_fwd", "cost_usd_expert_fwd",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)

def append_metrics(path: str, row: Dict[str, Any]):
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)


# ============================================================
# Real dataset (char LM)
# ============================================================

def _build_vocab(txt_path: str, vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(stoi, f, ensure_ascii=False, indent=2)
    itos = {int(i): ch for ch, i in stoi.items()}
    return stoi, itos

def _load_ids(txt_path: str, stoi: Dict[str, int]) -> torch.Tensor:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = [stoi.get(ch, 0) for ch in text]
    return torch.tensor(ids, dtype=torch.long)

class TextBatcher:
    def __init__(self, ids: torch.Tensor, batch_size: int, seq_len: int, seed: int = 42):
        self.ids = ids.contiguous()
        self.bs = int(batch_size)
        self.T = int(seq_len)
        self.rng = random.Random(seed)
        if self.ids.numel() < self.T + 2:
            raise ValueError("input.txt too short for seq_len")
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = int(self.ids.numel() - (self.T + 1))
        starts = [self.rng.randint(0, max_start) for _ in range(self.bs)]
        x = torch.stack([self.ids[s:s+self.T] for s in starts], dim=0)
        y = torch.stack([self.ids[s+1:s+1+self.T] for s in starts], dim=0)
        return x, y


# ============================================================
# Accuracy
# ============================================================

def _acc_topk(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=k, dim=-1).indices
    hit = (topk == y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return float(hit)


# ============================================================
# Async expert update policy
# ============================================================

class ExpertUpdatePolicy:
    def __init__(self, n: int, cold_acc: int):
        self.n = int(n)
        self.cold_acc = max(1, int(cold_acc))
        self.pending = [0 for _ in range(self.n)]

    def decide(self, hot_set: List[int]) -> Tuple[List[int], int, float]:
        if ABL_CFG.force_sync_update or FORCE_SYNC_UPDATE:
            self.pending = [0 for _ in range(self.n)]
            return list(range(self.n)), 0, 0.0
        upd = []
        cold_upd = 0
        for eid in range(self.n):
            if eid in hot_set:
                upd.append(eid)
                self.pending[eid] = 0
            else:
                self.pending[eid] += 1
                if self.pending[eid] >= self.cold_acc:
                    upd.append(eid)
                    cold_upd += 1
                    self.pending[eid] = 0
        return upd, cold_upd, float(np.mean(self.pending))

POLICY = ExpertUpdatePolicy(NUM_EXPERTS, COLD_ACC_STEPS)


# ============================================================
# One micro-batch step helpers
# ============================================================

def _get_candidates(func_name: str) -> List[Dict[str, Any]]:
    return [INST_BY_ID[i] for i in FUNC_MAP.get(func_name, []) if i in INST_BY_ID]

def _payload(obj: Any) -> dict:
    if isinstance(obj, dict) and "payload" in obj and isinstance(obj["payload"], dict):
        return obj["payload"]
    return obj if isinstance(obj, dict) else {}

def _to_tensor(v: Any, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        t = v
    elif isinstance(v, dict):
        try:
            t = pack_to_tensor(v)
        except Exception:
            t = torch.as_tensor(v)
    else:
        t = torch.as_tensor(v)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device)

async def _invoke_fn(
    func_name: str,
    *,
    mode: str,
    base_compute_ms: float,
    http_path: str,
    payload: Dict[str, Any],
    global_step: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Tuple[float,float,float,float,float], int]:
    cands = _get_candidates(func_name)
    if not cands:
        raise RuntimeError(f"FUNC_MAP has no candidates for {func_name}")

    inst, breakdown, retry_cnt = await invoke_with_retry(
        func_name, 0, cands, req={}, base_compute_ms=base_compute_ms,
        mode=mode, max_tries=INVOKE_RETRIES, global_step=global_step
    )

    if USE_HTTP_EXEC:
        obj = await invoke_http(inst, path=http_path, payload=payload)
    else:
        obj = {}

    return inst, obj, breakdown, retry_cnt


async def run_microbatch(global_step: int, mb_idx: int, x_tok: torch.Tensor, y_tok: torch.Tensor) -> Dict[str, Any]:
    """
    单个 micro-batch 的完整闭环：
      pre/fwd -> expert/fwd -> post/fwd -> post/bwd -> expert/bwd -> pre/bwd
    """
    metrics = {
        "pre_lat": 0.0, "post_lat": 0.0, "exp_lat": 0.0,
        "inv_total_ms": 0.0, "inv_queue_ms": 0.0, "inv_cold_ms": 0.0, "inv_net_ms": 0.0, "inv_compute_ms": 0.0,
        "inv_retry_cnt": 0.0,
        "cost_usd_pre_fwd": 0.0, "cost_usd_post_fwd": 0.0, "cost_usd_expert_fwd": 0.0,
        "fwd_mode_hot": 0.0, "fwd_mode_cold": 0.0, "fwd_mode_http": 0.0,
        "grad_mode_hot": 0.0, "grad_mode_cold": 0.0, "grad_mode_http": 0.0,
        "loss": float("nan"),
        "acc1": float("nan"),
        "acc5": float("nan"),
        "hot_ratio": 0.0,
        "hot_set_changed": 0.0,
        "hot_set_jaccard": 1.0,
    }

    dev = torch.device(DEVICE)

    # -------------------------
    # (1) pre fwd
    # -------------------------
    t0 = time.perf_counter()
    inst_pre, pre_obj, (tot, q, cold, net, comp), retry_cnt = await _invoke_fn(
        "moe.pre_fwd",
        mode="http",
        base_compute_ms=1.5,
        http_path=PATH_FWD,
        payload={"x": x_tok},
        global_step=global_step,
    )
    _ = (time.perf_counter() - t0) * 1000.0

    metrics["pre_lat"] += tot
    metrics["inv_total_ms"] += tot
    metrics["inv_queue_ms"] += q
    metrics["inv_cold_ms"] += cold
    metrics["inv_net_ms"] += net
    metrics["inv_compute_ms"] += comp
    metrics["inv_retry_cnt"] += retry_cnt
    metrics["cost_usd_pre_fwd"] += _cost_usd(inst_pre, tot)
    metrics["fwd_mode_http"] += 1

    if not USE_HTTP_EXEC:
        raise RuntimeError("USE_HTTP_EXEC=0 仅仿真延迟，不会做真实计算。你要真实计算请设 USE_HTTP_EXEC=1")

    pre_payload = _payload(pre_obj)
    h: torch.Tensor = _to_tensor(pre_payload["h"], device=dev)
    topk_vals: torch.Tensor = _to_tensor(pre_payload["topk_vals"], device=dev)
    topk_idx: torch.Tensor = _to_tensor(pre_payload["topk_idx"], device=dev, dtype=torch.long)

    # 更新热度统计
    try:
        async with HEATMAP_LOCK:
            HEATMAP.update_from_routing(topk_idx, topk_vals)
    except Exception:
        pass

    # ---------- Adaptive hot_set + mass-based hot_ratio + convergence metrics ----------
    HOT_COVERAGE = float(os.environ.get("HOT_COVERAGE", "0.70"))
    HOTSET_MIN = int(os.environ.get("HOTSET_MIN", "1"))
    HOTSET_MAX = int(os.environ.get("HOTSET_MAX", str(NUM_EXPERTS)))

    if ABL_CFG.disable_hotcold:
        hot_set = list(range(NUM_EXPERTS))
    else:
        scores = np.asarray(getattr(HEATMAP, "scores", np.ones(NUM_EXPERTS) / max(1, NUM_EXPERTS)), dtype=np.float32)
        order = np.argsort(-scores)
        cum = 0.0
        hot = []
        for e in order:
            hot.append(int(e))
            cum += float(scores[int(e)])
            if cum >= HOT_COVERAGE and len(hot) >= HOTSET_MIN:
                break
        if len(hot) < HOTSET_MIN:
            hot = list(map(int, order[:HOTSET_MIN]))
        if len(hot) > HOTSET_MAX:
            hot = list(map(int, order[:HOTSET_MAX]))
        hot_set = hot

    # hot_ratio：路由到热专家的权重 mass / 总权重 mass
    try:
        idx_flat = topk_idx.reshape(-1).detach().cpu().numpy()
        val_flat = topk_vals.reshape(-1).detach().cpu().numpy().astype(np.float32)
        if len(idx_flat) > 0:
            mask_hot = np.isin(idx_flat, np.asarray(hot_set, dtype=np.int64))
            hot_mass = float(val_flat[mask_hot].sum())
            tot_mass = float(val_flat.sum()) + 1e-12
            metrics["hot_ratio"] = hot_mass / tot_mass
        else:
            metrics["hot_ratio"] = 0.0
    except Exception:
        metrics["hot_ratio"] = 0.0

    global PREV_HOT_SET
    cur = set(hot_set)
    if PREV_HOT_SET is None:
        metrics["hot_set_changed"] = 0.0
        metrics["hot_set_jaccard"] = 1.0
    else:
        prev = set(PREV_HOT_SET)
        inter = len(cur & prev)
        union = max(1, len(cur | prev))
        metrics["hot_set_changed"] = float(1.0 if cur != prev else 0.0)
        metrics["hot_set_jaccard"] = float(inter / union)
    PREV_HOT_SET = list(hot_set)

    # -------------------------
    # (2) experts fwd (Top-K)
    # -------------------------
    B, T, D = h.shape
    combined = torch.zeros((B, T, D), dtype=h.dtype, device=h.device)
    route_cache: Dict[int, Dict[str, Any]] = {}

    idx_np = topk_idx.detach().cpu().numpy()
    selected = sorted(set(int(e) for e in np.unique(idx_np)))

    for e in selected:
        func_exp = f"moe.expert_fwd:{e}"
        if not _get_candidates(func_exp):
            continue

        mask = (idx_np == e)
        if not mask.any():
            continue

        b_idx, t_idx, k_idx = np.where(mask)
        inp = h[b_idx, t_idx, :]                         # [N,D]
        w = topk_vals[b_idx, t_idx, k_idx].unsqueeze(-1) # [N,1]

        mode = "http" if ABL_CFG.disable_hotcold else ("hot" if e in hot_set else "cold")
        base_ms = 2.0

        inst_exp, exp_obj, (tot, q, cold, net, comp), retry_cnt = await _invoke_fn(
            func_exp,
            mode=mode,
            base_compute_ms=base_ms,
            http_path=PATH_FWD,
            payload={"inp": inp, "eid": int(e)},
            global_step=global_step,
        )

        exp_payload = _payload(exp_obj)
        out = _to_tensor(exp_payload["out"], device=dev).to(h.device)

        combined[b_idx, t_idx, :] += out * w

        route_cache[e] = {
            "inst": inst_exp,
            "trace_id": exp_obj.get("trace_id"),
            "b_idx": b_idx.tolist(),
            "t_idx": t_idx.tolist(),
            "k_idx": k_idx.tolist(),
            "out": out.detach(),
        }

        metrics["exp_lat"] += tot
        metrics["inv_total_ms"] += tot
        metrics["inv_queue_ms"] += q
        metrics["inv_cold_ms"] += cold
        metrics["inv_net_ms"] += net
        metrics["inv_compute_ms"] += comp
        metrics["inv_retry_cnt"] += retry_cnt
        metrics["cost_usd_expert_fwd"] += _cost_usd(inst_exp, tot)

        n_assign = float(len(b_idx))
        if mode == "hot":
            metrics["fwd_mode_hot"] += n_assign
        elif mode == "cold":
            metrics["fwd_mode_cold"] += n_assign
        else:
            metrics["fwd_mode_http"] += n_assign

    # -------------------------
    # (3) post fwd
    # -------------------------
    inst_post, post_obj, (tot, q, cold, net, comp), retry_cnt = await _invoke_fn(
        "moe.post_fwd",
        mode="http",
        base_compute_ms=1.0,
        http_path=PATH_FWD,
        payload={"combined": combined, "y": y_tok},
        global_step=global_step,
    )

    metrics["post_lat"] += tot
    metrics["inv_total_ms"] += tot
    metrics["inv_queue_ms"] += q
    metrics["inv_cold_ms"] += cold
    metrics["inv_net_ms"] += net
    metrics["inv_compute_ms"] += comp
    metrics["inv_retry_cnt"] += retry_cnt
    metrics["cost_usd_post_fwd"] += _cost_usd(inst_post, tot)
    metrics["fwd_mode_http"] += 1

    post_payload = _payload(post_obj)
    loss_raw = post_payload.get("loss", None)
    logits_raw = post_payload.get("logits", None)

    try:
        if loss_raw is not None:
            loss_t = _to_tensor(loss_raw, device=dev).float()
            metrics["loss"] = float(loss_t.item())
    except Exception:
        pass

    try:
        if logits_raw is not None:
            logits_t = _to_tensor(logits_raw, device=dev).float()
            metrics["acc1"] = _acc_topk(logits_t, y_tok, k=1)
            metrics["acc5"] = _acc_topk(logits_t, y_tok, k=min(5, int(logits_t.size(-1))))
    except Exception:
        pass

    # -------------------------
    # (4) post bwd -> grad_combined
    # -------------------------
    post_bwd_obj = await invoke_http(
        inst_post,
        path=PATH_BWD,
        payload={"trace_id": post_obj.get("trace_id")},
    )
    if isinstance(post_bwd_obj, dict) and (post_bwd_obj.get("ok") is False):
        raise RuntimeError(f"post_fn /bwd ok=False, error={post_bwd_obj.get('error')}")

    post_bwd_payload = _payload(post_bwd_obj)
    grad_raw = None
    for k in ("grad_combined", "grad_h", "grad"):
        if k in post_bwd_payload and post_bwd_payload[k] is not None:
            grad_raw = post_bwd_payload[k]
            break
    if grad_raw is None:
        raise KeyError(f"post_fn /bwd missing grad. keys={list(post_bwd_payload.keys())}")

    grad_combined = _to_tensor(grad_raw, device=dev).to(h.device)

    # -------------------------
    # (5) expert bwd (stick to route_cache inst)
    # -------------------------
    grad_h = torch.zeros_like(h)
    grad_topk_vals = torch.zeros_like(topk_vals)

    update_eids, _cold_upd, _pending_mean = POLICY.decide(hot_set)

    for e in selected:
        if e not in route_cache:
            continue

        b_idx = route_cache[e]["b_idx"]
        t_idx = route_cache[e]["t_idx"]
        k_idx = route_cache[e]["k_idx"]
        out_vecs: torch.Tensor = route_cache[e]["out"]

        w = topk_vals[b_idx, t_idx, k_idx].unsqueeze(-1)
        grad_out = grad_combined[b_idx, t_idx, :] * w

        grad_w = (grad_combined[b_idx, t_idx, :] * out_vecs).sum(dim=-1)
        for i, (bi, ti, ki) in enumerate(zip(b_idx, t_idx, k_idx)):
            grad_topk_vals[bi, ti, ki] += grad_w[i]

        mode = "http" if ABL_CFG.disable_hotcold else ("hot" if e in hot_set else "cold")
        n_assign = float(len(b_idx))
        if mode == "hot":
            metrics["grad_mode_hot"] += n_assign
        elif mode == "cold":
            metrics["grad_mode_cold"] += n_assign
        else:
            metrics["grad_mode_http"] += n_assign

        inst_e = route_cache[e]["inst"]
        trace_id = route_cache[e]["trace_id"]

        exp_bwd_obj = await invoke_http(
            inst_e,
            path=PATH_BWD,
            payload={"trace_id": trace_id, "grad_out": grad_out},
        )
        if isinstance(exp_bwd_obj, dict) and (exp_bwd_obj.get("ok") is False):
            raise RuntimeError(f"expert {e} /bwd ok=False, error={exp_bwd_obj.get('error')}")

        grad_inp = _to_tensor(exp_bwd_obj.get("grad_inp"), device=dev).to(h.device)
        for i, (bi, ti) in enumerate(zip(b_idx, t_idx)):
            grad_h[bi, ti, :] += grad_inp[i]

    # -------------------------
    # (6) pre bwd (stick to inst_pre)
    # -------------------------
    pre_bwd_obj = await invoke_http(
        inst_pre,
        path=PATH_BWD,
        payload={
            "trace_id": pre_obj.get("trace_id"),
            "grad_h": grad_h,
            "grad_topk_vals": grad_topk_vals,
        },
    )
    if isinstance(pre_bwd_obj, dict) and (pre_bwd_obj.get("ok") is False):
        raise RuntimeError(f"pre_fn /bwd ok=False: {pre_bwd_obj}")

    # -------------------------
    # (7) expert step (hot immediate; cold accumulated)
    # -------------------------
    for e in selected:
        if e not in route_cache:
            continue
        do_step = (e in hot_set) or (e in update_eids)
        if not do_step:
            continue

        inst_e = route_cache[e]["inst"]
        scale = (1.0 / float(COLD_ACC_STEPS)) if (e not in hot_set and not (ABL_CFG.force_sync_update or FORCE_SYNC_UPDATE)) else 1.0
        step_obj = await invoke_http(inst_e, path=PATH_STEP, payload={"scale": scale})
        if isinstance(step_obj, dict) and (step_obj.get("ok") is False):
            raise RuntimeError(f"expert {e} /step ok=False: {step_obj}")

    return metrics


# ============================================================
# Train Loop
# ============================================================

async def train():
    if not os.path.exists(METRICS_FILE):
        write_metrics_header(METRICS_FILE)

    stoi, _ = _build_vocab(DATA_PATH, VOCAB_PATH)
    ids = _load_ids(DATA_PATH, stoi)

    n = int(ids.numel())
    n_train = max(SEQ_LEN + 2, int(n * 0.9))
    n_train = min(n - (SEQ_LEN + 2), n_train)
    train_ids = ids[:n_train].contiguous()
    val_ids = ids[n_train:].contiguous()

    train_batcher = TextBatcher(train_ids, BATCH_SIZE, SEQ_LEN, seed=SEED)
    val_batcher = TextBatcher(val_ids, BATCH_SIZE, SEQ_LEN, seed=SEED + 999)

    for step in range(1, MAX_STEPS + 1):
        t_step0 = time.perf_counter()

        split = "train"
        if (step % VAL_INTERVAL) == 0:
            split = "val"

        x, y = (val_batcher.next_batch() if split == "val" else train_batcher.next_batch())

        mb = max(1, MICRO_BATCH)
        xs = torch.chunk(x, mb, dim=0)
        ys = torch.chunk(y, mb, dim=0)

        rows = []
        for i in range(mb):
            r = await run_microbatch(step, i, xs[i], ys[i])
            rows.append(r)

        loss = float(np.mean([r["loss"] for r in rows]))
        acc1 = float(np.mean([r["acc1"] for r in rows]))
        acc5 = float(np.mean([r["acc5"] for r in rows]))
        hot_ratio = float(np.mean([r["hot_ratio"] for r in rows]))

        pre_lat = float(np.sum([r["pre_lat"] for r in rows]))
        post_lat = float(np.sum([r["post_lat"] for r in rows]))
        exp_lat = float(np.sum([r["exp_lat"] for r in rows]))

        inv_total_ms = float(np.sum([r["inv_total_ms"] for r in rows]))
        inv_queue_ms = float(np.sum([r["inv_queue_ms"] for r in rows]))
        inv_cold_ms = float(np.sum([r["inv_cold_ms"] for r in rows]))
        inv_net_ms = float(np.sum([r["inv_net_ms"] for r in rows]))
        inv_compute_ms = float(np.sum([r["inv_compute_ms"] for r in rows]))
        inv_retry_cnt = float(np.sum([r["inv_retry_cnt"] for r in rows]))

        fwd_hot = float(np.sum([r["fwd_mode_hot"] for r in rows]))
        fwd_cold = float(np.sum([r["fwd_mode_cold"] for r in rows]))
        fwd_http = float(np.sum([r["fwd_mode_http"] for r in rows]))
        fwd_total = max(1.0, fwd_hot + fwd_cold + fwd_http)

        grad_hot = float(np.sum([r["grad_mode_hot"] for r in rows]))
        grad_cold = float(np.sum([r["grad_mode_cold"] for r in rows]))
        grad_http = float(np.sum([r["grad_mode_http"] for r in rows]))
        grad_total = max(1.0, grad_hot + grad_cold + grad_http)

        cost_pre = float(np.sum([r["cost_usd_pre_fwd"] for r in rows]))
        cost_post = float(np.sum([r["cost_usd_post_fwd"] for r in rows]))
        cost_exp = float(np.sum([r["cost_usd_expert_fwd"] for r in rows]))
        cost_step = cost_pre + cost_post + cost_exp

        step_time_ms = (time.perf_counter() - t_step0) * 1000.0
        DEADLINE_EST.update(step_time_ms)

        deadline_ms = DEADLINE_EST.deadline_ms(step)
        viol = 1.0 if step_time_ms > deadline_ms else 0.0

        row = {
            "step": step, "split": split,
            "loss": loss, "acc_top1": acc1, "acc_top5": acc5,
            "step_time_ms": step_time_ms,
            "pre_lat_ms": pre_lat, "post_lat_ms": post_lat, "exp_lat_ms": exp_lat,
            "inv_total_ms": inv_total_ms, "inv_queue_ms": inv_queue_ms, "inv_cold_ms": inv_cold_ms,
            "inv_net_ms": inv_net_ms, "inv_compute_ms": inv_compute_ms,
            "inv_retry_cnt": inv_retry_cnt,
            "hot_ratio": hot_ratio,
            "fwd_mode_hot_frac": fwd_hot / fwd_total,
            "fwd_mode_cold_frac": fwd_cold / fwd_total,
            "fwd_mode_http_frac": fwd_http / fwd_total,
            "grad_mode_hot_frac": grad_hot / grad_total,
            "grad_mode_cold_frac": grad_cold / grad_total,
            "grad_mode_http_frac": grad_http / grad_total,
            "deadline_ms": deadline_ms,
            "deadline_violation_frac": viol,
            "cost_usd_step": cost_step,
            "cost_usd_pre_fwd": cost_pre,
            "cost_usd_post_fwd": cost_post,
            "cost_usd_expert_fwd": cost_exp,
            "hot_set_changed": float(np.mean([r.get("hot_set_changed", 0.0) for r in rows])),
            "hot_set_jaccard": float(np.mean([r.get("hot_set_jaccard", 1.0) for r in rows])),
        }

        append_metrics(METRICS_FILE, row)

        if (step % LOG_TRAIN_EVERY) == 0 or split == "val":
            print(
                f"[{split}] step={step} loss={loss:.4f} acc_top5={acc5:.4f} "
                f"step_ms={step_time_ms:.1f} cold_ms={inv_cold_ms:.1f} "
                f"hot_ratio={hot_ratio:.2f} "
                f"fwd(h/c/http)=({row['fwd_mode_hot_frac']:.2f}/{row['fwd_mode_cold_frac']:.2f}/{row['fwd_mode_http_frac']:.2f}) "
                f"grad(h/c/http)=({row['grad_mode_hot_frac']:.2f}/{row['grad_mode_cold_frac']:.2f}/{row['grad_mode_http_frac']:.2f})"
            )


def main():
    asyncio.run(train())


if __name__ == "__main__":
    main()
