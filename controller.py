# -------------------------------------------------------------------------
# [UPDATED] 在原 controller 基础上：
# 1) 保留你现有的 Ablation/Baseline + 调度 + 冷启动/网络/队列仿真 breakdown
# 2) 把“计算”改成真正的本地 serverless：HTTP 调用 pre_fn / expert_app / post_fn 多实例执行
# 3) 继续支持 Top-K 只调用被选择专家
# 4) 支持“热专家立即反传/更新，冷专家梯度累计，批量 step”（异步反传）
#
# [FIX] 关键修复：
# - 彻底改用 shared.py 的 tensor pack 协议（dumps/loads/tensor_to_pack/pack_to_tensor）
# - 请求体不再包一层 {"payload": ...}，避免服务端取值不一致
# - 自动兼容响应里带/不带 payload 两种格式
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

# ✅ 使用你项目既有的序列化协议（这一步是修复的核心）
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


def _pack_payload_with_shared(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把 Tensor 转成 shared.tensor_to_pack，其他类型原样。"""
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            out[k] = tensor_to_pack(v)
        else:
            out[k] = v
    return out


def _maybe_unpack_tensorpack(v: Any) -> Any:
    """如果 v 看起来像 tensor pack，则 pack_to_tensor；否则原样返回。"""
    if isinstance(v, dict):
        # shared.py 的 tensor pack 一般会包含 shape/dtype/data 或 bytes 等字段
        # 这里做“软判断”，失败就原样返回
        try:
            return pack_to_tensor(v)
        except Exception:
            return v
    return v


def _unpack_payload_with_shared(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        out[k] = _maybe_unpack_tensorpack(v)
    return out


async def _http_post_raw(url: str, path: str, body_bytes: bytes) -> bytes:
    full = url.rstrip("/") + path
    async with _HTTP_SEM:
        if aiohttp is not None:
            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(full, data=body_bytes) as resp:
                    txt = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status} {full}: {txt[:400]}")
                    return txt.encode("utf-8")
        else:
            def _do():
                r = requests.post(full, data=body_bytes, timeout=HTTP_TIMEOUT_S)
                r.raise_for_status()
                return r.content
            return await asyncio.to_thread(_do)


async def invoke_http(
    inst: Dict[str, Any],
    *,
    path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    最终稳定版：
    - /zero /step /health 也用 POST（因为你那边 GET 会 405）
    - 所有请求都用 application/json（解决你 POST /zero 的 422）
    """
    url = _inst_url(inst).rstrip("/")
    full = url + path

    # 1) 把 Tensor 转成 shared.tensor_to_pack 格式（JSON 可序列化）
    req_obj = _pack_payload_with_shared(payload)

    # 2) 强制 JSON：requests 的 json= 会自动带 Content-Type: application/json
    #    注意：req_obj 里已经没有 Tensor 了，所以可以直接 json=req_obj
    async with _HTTP_SEM:
        def _do():
            r = requests.post(full, json=req_obj, timeout=HTTP_TIMEOUT_S)
            r.raise_for_status()
            return r.content
        raw = await asyncio.to_thread(_do)

    if not raw:
        return {}

    obj = loads(raw)  # shared.loads: json->dict + unpack tensor pack

    # 兼容服务端返回 {"payload": {...}} 或直接 {...}
    if isinstance(obj, dict) and "payload" in obj and isinstance(obj["payload"], dict):
        obj = obj["payload"]

    return _unpack_payload_with_shared(obj)

# ============================================================
# Autoscaling (simple clone)
# ============================================================

AUTOSCALE_LAST: Dict[str, int] = {}


def _clone_inst_id(base_id: str, replica_idx: int) -> str:
    return f"{base_id}__r{replica_idx}"


def _maybe_autoscale(func_name: str, candidates: List[Dict[str, Any]], queue_ms: float, step: int):
    if not AUTOSCALE_ENABLE:
        return
    if queue_ms < AUTOSCALE_QUEUE_TH_MS:
        return
    last = AUTOSCALE_LAST.get(func_name, -999999)
    if (step - last) < AUTOSCALE_COOLDOWN_STEPS:
        return

    ids = FUNC_MAP.get(func_name, [])
    base_ids = [i for i in ids if "__r" not in i]
    if not base_ids:
        return
    base = base_ids[0]
    current = [i for i in ids if i.startswith(base)]
    if len(current) >= AUTOSCALE_MAX_REPLICA:
        return

    replica_idx = len(current)
    new_id = _clone_inst_id(base, replica_idx)
    if new_id in INST_BY_ID:
        return

    clone = json.loads(json.dumps(INST_BY_ID[base]))
    clone["id"] = new_id
    meta = clone.get("meta", {}) or {}
    meta["cold_start_ms"] = float(meta.get("cold_start_ms", 200.0)) * 0.6
    meta["performance"] = float(meta.get("performance", 1.0)) * 0.92
    clone["meta"] = meta

    INST_BY_ID[new_id] = clone
    _get_inst_sem(clone)

    FUNC_MAP.setdefault(func_name, [])
    if new_id not in FUNC_MAP[func_name]:
        FUNC_MAP[func_name].append(new_id)

    AUTOSCALE_LAST[func_name] = step


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
# Our hybrid scheduler (lightweight stats)
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
# Invocation wrapper with retry + baseline interception
# ============================================================

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
        inst = HYBRID_SCHED.select_best(cand) if not ABL_CFG.is_baseline else random.choice(cand)
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
        "hot_ratio", "fwd_mode_hot_frac", "fwd_mode_cold_frac", "fwd_mode_http_frac",
        "grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac",
        "deadline_ms", "deadline_violation_frac",
        "cost_usd_step", "cost_usd_pre_fwd", "cost_usd_post_fwd", "cost_usd_expert_fwd",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)

def append_metrics(path: str, row: Dict[str, Any]):
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)


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
# One micro-batch step
# ============================================================

def _get_candidates(func_name: str) -> List[Dict[str, Any]]:
    return [INST_BY_ID[i] for i in FUNC_MAP.get(func_name, []) if i in INST_BY_ID]

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
    B, T = x_tok.shape

    metrics = {
        "pre_lat": 0.0, "post_lat": 0.0, "exp_lat": 0.0,
        "inv_total_ms": 0.0, "inv_queue_ms": 0.0, "inv_cold_ms": 0.0, "inv_net_ms": 0.0, "inv_compute_ms": 0.0,
        "inv_retry_cnt": 0.0,
        "cost_usd_pre_fwd": 0.0, "cost_usd_post_fwd": 0.0, "cost_usd_expert_fwd": 0.0,
        "fwd_mode_hot": 0, "fwd_mode_cold": 0, "fwd_mode_http": 0,
        "grad_mode_hot": 0, "grad_mode_cold": 0, "grad_mode_http": 0,
        "loss": float("nan"),
        "acc1": float("nan"),
        "acc5": float("nan"),
        "hot_ratio": 0.0,
    }

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
    real_pre_ms = (time.perf_counter() - t0) * 1000.0

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
        raise RuntimeError("USE_HTTP_EXEC=0 仅仿真延迟，不会做真实计算。你要拆分实例执行请设 USE_HTTP_EXEC=1")

    h: torch.Tensor = pre_obj["h"]
    topk_vals: torch.Tensor = pre_obj["topk_vals"]
    topk_idx: torch.Tensor = pre_obj["topk_idx"]

    if TRAFFIC_SKEW_ENABLE:
        topk_idx, topk_vals = simulate_traffic_skew(topk_idx, topk_vals, NUM_EXPERTS, global_step=global_step)

    if not ABL_CFG.disable_hotcold:
        async with HEATMAP_LOCK:
            HEATMAP.update_from_routing(topk_idx, topk_vals)

    hot_set = HEATMAP.hot_set(HOTSET_SIZE) if (not ABL_CFG.disable_hotcold) else list(range(NUM_EXPERTS))

    idx_np = topk_idx.detach().cpu().numpy()
    val_np = topk_vals.detach().cpu().numpy()
    denom = float(val_np.sum() + 1e-12)
    hot_mass = float(val_np[np.isin(idx_np, np.array(hot_set))].sum())
    metrics["hot_ratio"] = float(hot_mass / denom)

    # -------------------------
    # (2) experts fwd (Top-K)
    # -------------------------
    B, T, D = h.shape
    combined = torch.zeros((B, T, D), dtype=h.dtype)

    route_out_cache: Dict[int, List[Tuple[int,int,int, torch.Tensor]]] = {i: [] for i in range(NUM_EXPERTS)}
    selected = sorted(set(int(e) for e in np.unique(idx_np)))

    for e in selected:
        func_exp = f"moe.expert_fwd:{e}"
        if not _get_candidates(func_exp):
            continue

        mask = (idx_np == e)
        if not mask.any():
            continue

        b_idx, t_idx, k_idx = np.where(mask)
        inp = h[b_idx, t_idx, :]  # [N,D]
        w = topk_vals[b_idx, t_idx, k_idx].unsqueeze(-1)  # [N,1]

        mode = "http" if ABL_CFG.disable_hotcold else ("hot" if e in hot_set else "cold")
        load = float((topk_idx == e).float().mean().item())
        base_ms = max(1.0, real_pre_ms * (0.6 + 1.2 * load))

        inst_exp, exp_obj, (tot, q, cold, net, comp), retry_cnt = await _invoke_fn(
            func_exp,
            mode=mode,
            base_compute_ms=base_ms,
            http_path=PATH_FWD,
            payload={"inp": inp, "eid": int(e)},
            global_step=global_step,
        )

        out: torch.Tensor = exp_obj["out"]
        combined[b_idx, t_idx, :] += out * w

        for bi, ti, ki, ov in zip(b_idx.tolist(), t_idx.tolist(), k_idx.tolist(), out):
            route_out_cache[e].append((bi, ti, ki, ov.detach().clone()))

        metrics["exp_lat"] += tot
        metrics["inv_total_ms"] += tot
        metrics["inv_queue_ms"] += q
        metrics["inv_cold_ms"] += cold
        metrics["inv_net_ms"] += net
        metrics["inv_compute_ms"] += comp
        metrics["inv_retry_cnt"] += retry_cnt
        metrics["cost_usd_expert_fwd"] += _cost_usd(inst_exp, tot)

        if mode == "hot":
            metrics["fwd_mode_hot"] += 1
        elif mode == "cold":
            metrics["fwd_mode_cold"] += 1
        else:
            metrics["fwd_mode_http"] += 1

    # -------------------------
    # (3) post fwd
    # -------------------------
    t1 = time.perf_counter()
    inst_post, post_obj, (tot, q, cold, net, comp), retry_cnt = await _invoke_fn(
        "moe.post_fwd",
        mode="http",
        base_compute_ms=max(1.0, real_pre_ms * 0.25),
        http_path=PATH_FWD,
        payload={"combined": combined, "y": y_tok},
        global_step=global_step,
    )
    _ = (time.perf_counter() - t1) * 1000.0

    metrics["post_lat"] += tot
    metrics["inv_total_ms"] += tot
    metrics["inv_queue_ms"] += q
    metrics["inv_cold_ms"] += cold
    metrics["inv_net_ms"] += net
    metrics["inv_compute_ms"] += comp
    metrics["inv_retry_cnt"] += retry_cnt
    metrics["cost_usd_post_fwd"] += _cost_usd(inst_post, tot)

    loss_t: torch.Tensor = post_obj["loss"]
    logits: Optional[torch.Tensor] = post_obj.get("logits", None)

    metrics["loss"] = float(loss_t.item())
    if isinstance(logits, torch.Tensor):
        metrics["acc1"] = _acc_topk(logits, y_tok, k=1)
        metrics["acc5"] = _acc_topk(logits, y_tok, k=min(5, logits.size(-1)))

    # -------------------------
    # (4) post bwd -> grad_combined
    # -------------------------
    _, post_bwd_obj, _, _ = await _invoke_fn(
        "moe.post_fwd", mode="http", base_compute_ms=1.0, http_path=PATH_BWD, payload={}, global_step=global_step
    )
    grad_combined: torch.Tensor = post_bwd_obj["grad_combined"]

    # -------------------------
    # (5) expert bwd
    # -------------------------
    grad_h = torch.zeros_like(h)
    grad_topk_vals = torch.zeros_like(topk_vals)

    update_eids, _, _ = POLICY.decide(hot_set)

    for e in selected:
        do_step = (e in hot_set) or (e in update_eids)
        if do_step:
            func_exp = f"moe.expert_fwd:{e}"
            await _invoke_fn(func_exp, mode="http", base_compute_ms=1.0, http_path=PATH_ZERO, payload={}, global_step=global_step)

    for e, routes in route_out_cache.items():
        if not routes:
            continue

        b_idx = [r[0] for r in routes]
        t_idx = [r[1] for r in routes]
        k_idx = [r[2] for r in routes]
        out_vecs = torch.stack([r[3] for r in routes], dim=0)

        w = topk_vals[b_idx, t_idx, k_idx].unsqueeze(-1)
        grad_out = grad_combined[b_idx, t_idx, :] * w

        grad_w = (grad_combined[b_idx, t_idx, :] * out_vecs).sum(dim=-1)
        for i, (bi, ti, ki) in enumerate(zip(b_idx, t_idx, k_idx)):
            grad_topk_vals[bi, ti, ki] += grad_w[i]

        func_exp = f"moe.expert_fwd:{e}"
        mode = "http" if ABL_CFG.disable_hotcold else ("hot" if e in hot_set else "cold")

        if mode == "hot":
            metrics["grad_mode_hot"] += 1
        elif mode == "cold":
            metrics["grad_mode_cold"] += 1
        else:
            metrics["grad_mode_http"] += 1

        _, exp_bwd_obj, _, _ = await _invoke_fn(
            func_exp, mode=mode, base_compute_ms=1.0, http_path=PATH_BWD, payload={"grad_out": grad_out}, global_step=global_step
        )
        grad_inp: torch.Tensor = exp_bwd_obj["grad_inp"]

        for i, (bi, ti) in enumerate(zip(b_idx, t_idx)):
            grad_h[bi, ti, :] += grad_inp[i]

    # -------------------------
    # (6) pre bwd
    # -------------------------
    await _invoke_fn(
        "moe.pre_fwd", mode="http", base_compute_ms=1.0, http_path=PATH_BWD,
        payload={"grad_h": grad_h, "grad_topk_vals": grad_topk_vals}, global_step=global_step
    )

    # -------------------------
    # (7) step
    # -------------------------
    await _invoke_fn("moe.pre_fwd", mode="http", base_compute_ms=1.0, http_path=PATH_STEP, payload={}, global_step=global_step)
    await _invoke_fn("moe.post_fwd", mode="http", base_compute_ms=1.0, http_path=PATH_STEP, payload={}, global_step=global_step)

    for e in selected:
        do_step = (e in hot_set) or (e in update_eids)
        if do_step:
            func_exp = f"moe.expert_fwd:{e}"
            scale = (1.0 / float(COLD_ACC_STEPS)) if (e not in hot_set and not (ABL_CFG.force_sync_update or FORCE_SYNC_UPDATE)) else 1.0
            await _invoke_fn(func_exp, mode="http", base_compute_ms=1.0, http_path=PATH_STEP, payload={"scale": scale}, global_step=global_step)

    return {
        "loss": metrics["loss"],
        "acc1": metrics["acc1"],
        "acc5": metrics["acc5"],
        "hot_ratio": metrics["hot_ratio"],
        "fwd_mode_hot": metrics["fwd_mode_hot"],
        "fwd_mode_cold": metrics["fwd_mode_cold"],
        "fwd_mode_http": metrics["fwd_mode_http"],
        "grad_mode_hot": metrics["grad_mode_hot"],
        "grad_mode_cold": metrics["grad_mode_cold"],
        "grad_mode_http": metrics["grad_mode_http"],
        "pre_lat": metrics["pre_lat"],
        "post_lat": metrics["post_lat"],
        "exp_lat": metrics["exp_lat"],
        "inv_total_ms": metrics["inv_total_ms"],
        "inv_queue_ms": metrics["inv_queue_ms"],
        "inv_cold_ms": metrics["inv_cold_ms"],
        "inv_net_ms": metrics["inv_net_ms"],
        "inv_compute_ms": metrics["inv_compute_ms"],
        "inv_retry_cnt": metrics["inv_retry_cnt"],
        "cost_usd_pre_fwd": metrics["cost_usd_pre_fwd"],
        "cost_usd_post_fwd": metrics["cost_usd_post_fwd"],
        "cost_usd_expert_fwd": metrics["cost_usd_expert_fwd"],
    }


# ============================================================
# Main training loop
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
        }

        append_metrics(METRICS_FILE, row)

        if (step % LOG_TRAIN_EVERY) == 0 or split == "val":
            print(
                f"[{split}] step={step} loss={loss:.4f} acc5={acc5:.4f} "
                f"step_ms={step_time_ms:.1f} cold_ms={inv_cold_ms:.1f} "
                f"hot_ratio={hot_ratio:.2f} fwd(h/c/http)=({row['fwd_mode_hot_frac']:.2f}/{row['fwd_mode_cold_frac']:.2f}/{row['fwd_mode_http_frac']:.2f}) "
                f"grad(h/c/http)=({row['grad_mode_hot_frac']:.2f}/{row['grad_mode_cold_frac']:.2f}/{row['grad_mode_http_frac']:.2f})"
            )


def main():
    asyncio.run(train())


if __name__ == "__main__":
    main()
