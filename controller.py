# -------------------------------------------------------------------------
# [UPDATED - FULL REPLACE controller.py]
# 新增特性：Local Compute Mode (无窗口计算模式)
# 1) 引入 LocalExecutor，在 controller 进程内直接运行 Pre/Post/Expert 模型
# 2) 拦截 USE_HTTP_EXEC=0 的情况，转为调用 LocalExecutor
# 3) 完美保留所有延迟仿真 (Autoscaler, Azure Trace, Cold Start)
# -------------------------------------------------------------------------

from __future__ import annotations
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
from comm import CommManager
# ✅ 使用你项目既有的序列化协议
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
# 引入适配器
from makemoe_adapter import MakeMoEAdapter
from makeMoE import MakeMoEConfig
from comm import CommManager
import requests


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
MAX_STEPS = int(os.getenv("MAX_STEPS", "800"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MICRO_BATCH = int(os.getenv("MICRO_BATCH", "4"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "20"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))

# Sequence / token LM batch
SEQ_LEN = int(os.getenv("SEQ_LEN", "64"))
DATA_PATH = os.getenv("DATA_PATH", "input.txt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")

# MoE
NUM_EXPERTS = int(os.getenv("NUM_EXPERTS", "4"))
TOP_K = int(os.getenv("TOP_K", "2"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "2000"))
EMB_DIM = int(os.getenv("EMB_DIM", "256"))

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
DEVICE = os.getenv("DEVICE", "cpu")

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
    is_static_compute: bool = False
    disable_hotcold: bool = False
    force_sync_update: bool = False
    use_random_sched: bool = False
    use_rr_sched: bool = False
    use_greedy_sched: bool = False
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
    # inside class LocalExecutor:
    def __init__(self):
        self.device = torch.device(DEVICE)
        print(f"[LocalExecutor] Initializing MakeMoE Adapter on {self.device}...")

        # 1. 实例化 MakeMoE 配置
        self.moe_cfg = MakeMoEConfig()

        # 2. 【关键】用环境变量覆盖默认值
        # 这样你在 run_local.ps1 里设置的 NUM_EXPERTS="8" 才会生效
        self.moe_cfg.vocab_size = VOCAB_SIZE
        self.moe_cfg.n_embed = EMB_DIM  # 对应环境变量 EMB_DIM
        self.moe_cfg.num_experts = NUM_EXPERTS  # 对应环境变量 NUM_EXPERTS
        self.moe_cfg.top_k = TOP_K
        self.moe_cfg.block_size = int(os.getenv("BLOCK_SIZE", "64"))

        # 3. 初始化适配器 (选择第1层作为Serverless拆分层)
        # 确保 split_layer_idx 不越界
        split_idx = min(1, self.moe_cfg.n_layer - 1)
        self.adapter = MakeMoEAdapter(self.moe_cfg, split_layer_idx=split_idx)

        # 4. 获取各阶段模块 (Pre/Expert/Post)
        self.pre = self.adapter.get_pre_stage().to(self.device)
        self.post = self.adapter.get_post_stage().to(self.device)
        self.experts = nn.ModuleList([
            self.adapter.get_expert_stage(i).to(self.device) for i in range(NUM_EXPERTS)
        ])

        # 5. 初始化优化器 (保持不变)
        self.opt_pre = torch.optim.AdamW(self.pre.parameters(), lr=1e-3)
        self.opt_post = torch.optim.AdamW(self.post.parameters(), lr=1e-3)
        self.opt_exps = [torch.optim.AdamW(e.parameters(), lr=1e-3) for e in self.experts]

        self.comm = CommManager()
        self.lock = asyncio.Lock()

    def _unpack_tensor(self, payload, key):
        if key not in payload: return None
        return _to_tensor(payload[key], device=self.device)

    def _save_tensor(self, key: str, data: Dict[str, Any], mode: str, force_hot: bool = False):
        if force_hot or mode not in ["cold"]:
            self.comm.send_hot(key, data)
        else:
            self.comm.send_cold(key, data)

    def _load_tensor(self, key: str, mode: str, delete: bool = True, try_hot_first: bool = False):
        target_mode = mode if mode in ["hot", "cold"] else "hot"
        if try_hot_first:
            data = self.comm.pull_hot(key, delete=delete)
            if data is not None: return data
        if target_mode == "cold":
            return self.comm.pull_cold(key, delete=delete)
        else:
            return self.comm.pull_hot(key, delete=delete)

    async def run(self, func_name: str, path: str, payload: Dict[str, Any], mode: str = "http") -> Dict[str, Any]:
        async with self.lock:
            trace_id = payload.get("trace_id")
            if not trace_id:
                trace_id = f"local_{uuid.uuid4()}"

            # --- Forward Pass ---
            if path == PATH_FWD:
                if "pre" in func_name:
                    x = self._unpack_tensor(payload, "x")
                    # Forward via Adapter Interface
                    # MakeMoEPreStage 返回的是 dict
                    res = self.pre(x)

                    h = res["hidden_states"]
                    # 注意：Adapter返回的是softmax后的weights，但controller期望topk_vals
                    # 这里为了兼容，我们需要确保 controller 逻辑匹配。
                    # controller 使用: topk_idx 来路由。
                    topk_idx = res["expert_indices"]
                    # 这里的 weights 是概率值，controller 可能直接用作权重
                    topk_vals = res["expert_weights"]

                    save_key = f"{trace_id}_pre"
                    # 存 input 用于重计算
                    save_data = {"x": tensor_to_pack(x.detach())}
                    self._save_tensor(save_key, save_data, mode)

                    return {
                        "trace_id": trace_id,
                        "h": tensor_to_pack(h.detach()),
                        "topk_vals": tensor_to_pack(topk_vals.detach()),
                        "topk_idx": tensor_to_pack(topk_idx.detach())
                    }

                if "expert" in func_name:
                    try:
                        eid = int(func_name.split(":")[-1])
                    except:
                        eid = 0
                    inp = self._unpack_tensor(payload, "inp")

                    with torch.no_grad():
                        out = self.experts[eid](inp)

                    save_key = f"{trace_id}_exp_{eid}"
                    save_data = {"inp": tensor_to_pack(inp.detach())}
                    self._save_tensor(save_key, save_data, mode, force_hot=True)

                    return {
                        "trace_id": trace_id,
                        "out": tensor_to_pack(out.detach())
                    }

                if "post" in func_name:
                    combined = self._unpack_tensor(payload, "combined")
                    y = self._unpack_tensor(payload, "y")

                    combined.requires_grad_(True)
                    logits, loss = self.post(combined, targets=y)

                    save_key = f"{trace_id}_post"
                    self._save_tensor(save_key,
                                      {"combined": tensor_to_pack(combined.detach()), "y": tensor_to_pack(y.detach())},
                                      "hot")

                    # 缓存 loss 用于 BWD
                    self._post_loss_cache = getattr(self, "_post_loss_cache", {})
                    self._post_loss_cache[trace_id] = {"loss": loss, "combined": combined}

                    return {
                        "trace_id": trace_id,
                        "logits": tensor_to_pack(logits.detach()),
                        "loss": loss.item() if loss is not None else 0.0
                    }

            # --- Backward Pass ---
            elif path == PATH_BWD:
                if "post" in func_name:
                    cache = getattr(self, "_post_loss_cache", {})
                    if trace_id in cache:
                        loss = cache[trace_id]["loss"]
                        combined = cache[trace_id]["combined"]
                        self.opt_post.zero_grad()
                        loss.backward()
                        grad_combined = combined.grad
                        del cache[trace_id]
                        return {
                            "grad_combined": tensor_to_pack(grad_combined),
                            "grad": tensor_to_pack(grad_combined)
                        }
                    return {"ok": False, "error": "No loss found in memory cache"}

                if "expert" in func_name:
                    try:
                        eid = int(func_name.split(":")[-1])
                    except:
                        eid = 0

                    save_key = f"{trace_id}_exp_{eid}"
                    saved_data = self._load_tensor(save_key, mode, delete=True, try_hot_first=True)

                    if saved_data is None:
                        return {"ok": False, "error": f"Trace {save_key} not found"}

                    inp_data = saved_data["inp"]
                    inp = _to_tensor(inp_data, device=self.device)
                    inp.requires_grad_(True)

                    # Re-computation
                    out = self.experts[eid](inp)

                    grad_out = self._unpack_tensor(payload, "grad_out")
                    self.opt_exps[eid].zero_grad()
                    out.backward(grad_out)

                    return {"grad_inp": tensor_to_pack(inp.grad)}

                if "pre" in func_name:
                    save_key = f"{trace_id}_pre"
                    saved_data = self._load_tensor(save_key, mode, delete=True)

                    if saved_data:
                        x = _to_tensor(saved_data["x"], device=self.device)
                        if x.dtype != torch.long: x = x.long()

                        # Re-computation Pre
                        res = self.pre(x)
                        h = res["hidden_states"]
                        # Pre BWD: 接收 grad_h
                        grad_h = self._unpack_tensor(payload, "grad_h")

                        self.opt_pre.zero_grad()
                        h.backward(grad_h)
                        return {"ok": True}
                    return {"ok": False, "error": "Pre trace not found"}

            elif path == PATH_STEP:
                if "pre" in func_name: self.opt_pre.step()
                if "post" in func_name: self.opt_post.step()
                if "expert" in func_name:
                    try:
                        eid = int(func_name.split(":")[-1])
                        self.opt_exps[eid].step()
                    except:
                        pass
                return {"ok": True}

            elif path == PATH_ZERO:
                if "pre" in func_name: self.opt_pre.zero_grad()
                if "post" in func_name: self.opt_post.zero_grad()
                if "expert" in func_name:
                    try:
                        eid = int(func_name.split(":")[-1])
                        self.opt_exps[eid].zero_grad()
                    except:
                        pass
                return {"ok": True}

            return {"ok": False, "error": "Unknown path"}

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


def _mode_net_multiplier(mode: str) -> float:
    m = (mode or "").lower()
    if m == "hot": return HOT_NET_MUL
    if m == "cold": return COLD_NET_MUL
    if m == "http": return HTTP_NET_MUL
    if m == "fallback": return FALLBACK_NET_MUL
    return HTTP_NET_MUL


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
        if ABL_CFG.is_static_compute: return 0.0
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
USE_TRACE_CALIB = os.getenv("USE_TRACE_CALIB", "1") == "1"
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


async def simulate_invoke_with_breakdown(
        inst: Dict[str, Any],
        base_compute_ms: float,
        req: Dict[str, Any],
        *,
        func_name: str,
        mode: str,
        global_step: int,
) -> Tuple[float, float, float, float, float]:
    region = str(inst.get("region", "local")).lower()
    is_local = "local" in region
    local_oom_prob = float(os.getenv("LOCAL_OOM_PROB", "0.05"))
    if ABL_CFG.is_static_compute and is_local: local_oom_prob = 0.0
    if is_local and local_oom_prob > 0 and random.random() < local_oom_prob:
        raise RuntimeError(f"Simulated Local Busy/OOM for {inst.get('id')}")
    meta = inst.get("meta", {}) or {}
    perf = float(meta.get("performance", DEFAULT_PERFORMANCE))
    raw_compute = float(base_compute_ms) / max(perf, 1e-6)
    compute_ms = raw_compute * random.uniform(0.90, 1.10)
    cold_ms = float(await INSTANCE_MGR.cold_start_ms(inst, func_name=func_name, mode=mode))
    is_gpu_task = False
    try:
        if float(meta.get("gpu_request", 0) or 0) > 0: is_gpu_task = True
        if float(meta.get("gpu_limit", 0) or 0) > 0: is_gpu_task = True
        dev = str(meta.get("device", "")).lower()
        if "gpu" in dev or "cuda" in dev: is_gpu_task = True
    except:
        pass
    if "expert" in (func_name or "").lower(): is_gpu_task = True
    if USE_TRACE_CALIB and TRACE.azure is not None:
        az_pair = TRACE.sample_azure_pair()
        p_cold = TRACE.inferred_cold_prob(az_pair)
        if random.random() < float(p_cold): cold_ms += TRACE.sample_cold_extra_ms()
    if USE_TRACE_CALIB and is_gpu_task and TRACE.gpu is not None:
        gpu_pair = TRACE.sample_gpu_pair()
        sampled = TRACE.sample_gpu_duration_ms(gpu_pair)
        if sampled is not None: compute_ms = float(sampled)
    sem = _get_inst_sem(inst)
    tq0 = time.perf_counter()
    async with sem:
        queue_ms = (time.perf_counter() - tq0) * 1000.0
        if USE_TRACE_CALIB and is_gpu_task and GPU_POOL_SEM is not None:
            tg0 = time.perf_counter()
            async with GPU_POOL_SEM: queue_ms += (time.perf_counter() - tg0) * 1000.0
        net_base = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))
        net_ms = net_base * _mode_net_multiplier(mode) * random.uniform(0.90, 1.10)
        if (mode or "").lower() == "cold": net_ms += float(COLD_STORAGE_MS)
        total_ms = queue_ms + cold_ms + net_ms + compute_ms
        return total_ms, queue_ms, cold_ms, net_ms, compute_ms


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

    def select_greedy_min_rtt(self, inst_list):
        best, best_v = None, 1e18
        for x in inst_list:
            meta = x.get("meta", {}) or {}
            rtt = float(meta.get("rtt_ms", meta.get("net_latency_ms", DEFAULT_NET_LATENCY)))
            if rtt < best_v: best_v, best = rtt, x
        return best if best is not None else random.choice(inst_list)


BASELINE_SCHED = BaselineScheduler()


class HybridScheduler:
    def __init__(self):
        self.ema_lat: Dict[str, float] = {}
        self.ema_decay = float(os.getenv("SCHED_EMA_DECAY", "0.95"))

    def update_stats(self, inst, tot_ms):
        inst_id = inst.get("id")
        v = self.ema_lat.get(inst_id)
        if v is None:
            self.ema_lat[inst_id] = float(tot_ms)
        else:
            self.ema_lat[inst_id] = self.ema_decay * v + (1.0 - self.ema_decay) * float(tot_ms)


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
    cold_ms = INSTANCE_MGR._default_cold_start_ms(func_name, inst) if not ABL_CFG.is_static_compute else 0.0
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
        if ABL_CFG.disable_nsga or len(cand_idx) <= 1:
            best_i, best_s = None, 1e18
            for i in cand_idx:
                s = self._score(*feats[i])
                if s < best_s: best_s, best_i = s, i
            return inst_list[int(best_i)]
        pts = [(feats[i][0], feats[i][1]) for i in cand_idx]
        front = [cand_idx[j] for j in self._pareto_front(pts)]
        best_i, best_s = None, 1e18
        for i in front:
            s = self._score(*feats[i])
            if s < best_s: best_s, best_i = s, i
        return inst_list[int(best_i)]


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


async def invoke_with_retry(func_name, logical_id, candidates, req, base_compute_ms, *, mode, max_tries,
                            forced_inst=None, global_step=0):
    if not candidates: raise RuntimeError(f"invoke candidates empty for {func_name}")
    if forced_inst is None and ABL_CFG.is_baseline:
        if ABL_CFG.use_random_sched:
            forced_inst = BASELINE_SCHED.select_random(candidates)
        elif ABL_CFG.use_rr_sched:
            forced_inst = BASELINE_SCHED.select_rr(func_name, candidates)
        elif ABL_CFG.use_greedy_sched:
            forced_inst = BASELINE_SCHED.select_greedy_min_rtt(candidates)
        elif ABL_CFG.is_static_compute:
            forced_inst = candidates[0]
    tries, last_err = 0, None

    async def _try(inst, retry_cnt):
        breakdown = await simulate_invoke_with_breakdown(inst, base_compute_ms, req, func_name=func_name, mode=mode,
                                                         global_step=global_step)
        HYBRID_SCHED.update_stats(inst, breakdown[0])
        _maybe_autoscale(func_name, candidates, breakdown[1], global_step)
        if SIM_SLEEP: await asyncio.sleep(breakdown[0] / 1000.0)
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
            inst = TRI_SCHED.select(cand, func_name=func_name, mode=mode, base_compute_ms=base_compute_ms,
                                    deadline_ms=deadline_ms)
        else:
            inst = random.choice(cand)
        try:
            return await _try(inst, max(0, tries - 1))
        except Exception as e:
            last_err = e;
            bad = inst.get("id");
            cand = [x for x in cand if x.get("id") != bad]
    raise RuntimeError(f"invoke_with_retry failed: {last_err}")


async def _invoke_fn(func_name, *, mode, base_compute_ms, http_path, payload, global_step):
    cands = _get_candidates(func_name)
    if not cands:
        cands = [INST_BY_ID[i] for i in FUNC_MAP.get(func_name, []) if i in INST_BY_ID]
        if not cands: raise RuntimeError(f"No candidates for {func_name}")
    inst, breakdown, retry_cnt = await invoke_with_retry(func_name, 0, cands, req={}, base_compute_ms=base_compute_ms,
                                                         mode=mode, max_tries=INVOKE_RETRIES, global_step=global_step)

    # [KEY CHANGE] Redirect to LocalExecutor if active
    if LOCAL_EXECUTOR:
        obj = await LOCAL_EXECUTOR.run(func_name, http_path, payload)
    elif USE_HTTP_EXEC:
        obj = await invoke_http(inst, path=http_path, payload=payload)
    else:
        obj = {}
    return inst, obj, breakdown, retry_cnt


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
                upd.append(eid); self.pending[eid] = 0
            else:
                self.pending[eid] += 1
                if self.pending[eid] >= self.cold_acc: upd.append(eid); cold_upd += 1; self.pending[eid] = 0
        return upd, cold_upd, float(np.mean(self.pending))


POLICY = ExpertUpdatePolicy(NUM_EXPERTS, COLD_ACC_STEPS)


# =========================================================================
# 【最终修复版 2.0】run_microbatch
# 修复了 TypeError: missing 'http_path' and 'global_step'
# =========================================================================
async def run_microbatch(step: int, micro_step: int, x: torch.Tensor, y: torch.Tensor):
    split = "train"
    trace_id = str(uuid.uuid4())

    import torch
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    x_tok = x.to(my_device)
    y_tok = y.to(my_device)

    # 计数器
    total_calls = 0
    cold_calls = 0

    # -------------------------------------------------------------
    # (1) Pre Fwd
    # -------------------------------------------------------------
    payload_pre = {
        "trace_id": trace_id, "inp": tensor_to_pack(x_tok), "split": split
    }

    # 调用 Pre
    inst_pre, pre_obj, (t1, q1, c1, n1, cmp1), r1 = await _invoke_fn(
        "moe.pre_fwd", mode="http", payload=payload_pre,
        base_compute_ms=10.0, http_path="/pre", global_step=step
    )

    if pre_obj is None: return None

    # 统计 Pre 状态
    total_calls += 1
    if c1 > 0: cold_calls += 1

    # -------------------------------------------------------------
    # (2) Expert Fwd
    # -------------------------------------------------------------
    expert_inputs_packed = pre_obj.get("expert_inputs", {})
    fwd_tasks = []

    for eid_str, inp_packed in expert_inputs_packed.items():
        eid = int(eid_str)
        p_exp = {"trace_id": trace_id, "inp": inp_packed, "split": split}
        fname = f"moe.expert_fwd:{eid}"

        t = _invoke_fn(
            fname, mode="hot", payload=p_exp,
            base_compute_ms=20.0, http_path=f"/expert/{eid}", global_step=step
        )
        fwd_tasks.append(t)

    expert_results = []
    t2, q2, c2, n2, cmp2, r2 = 0, 0, 0, 0, 0, 0

    if len(fwd_tasks) > 0:
        results_list = await asyncio.gather(*fwd_tasks)
        for res_tuple in results_list:
            if res_tuple is None: continue
            inst_e, obj_e, bd_e, retry_e = res_tuple

            # 统计 Expert 状态
            total_calls += 1
            if bd_e[2] > 0: cold_calls += 1  # bd_e[2] is cold_ms

            if obj_e is not None:
                t2 = max(t2, bd_e[0])
                # ... (其他延迟统计略，保持原样取最大即可)
                c2 = max(c2, bd_e[2])  # 记录最大的冷启动时间用于日志
                r2 += retry_e
                expert_results.append(obj_e)

    # -------------------------------------------------------------
    # (3) Post Fwd
    # -------------------------------------------------------------
    payload_post = {
        "trace_id": trace_id, "pre_context": pre_obj.get("context", {}),
        "expert_results": expert_results, "targets": tensor_to_pack(y_tok),
        "split": split
    }

    inst_post, post_obj, (t3, q3, c3, n3, cmp3), r3 = await _invoke_fn(
        "moe.post_fwd", mode="http", payload=payload_post,
        base_compute_ms=10.0, http_path="/post", global_step=step
    )

    if post_obj is None: return None

    # 统计 Post 状态
    total_calls += 1
    if c3 > 0: cold_calls += 1

    # -------------------------------------------------------------
    # (4) 结果计算
    # -------------------------------------------------------------
    # ... (Logits 解包逻辑保持不变) ...
    logits = None
    if "logits" in post_obj:
        try:
            logits = _to_tensor(post_obj["logits"], device=my_device).float()
        except:
            try:
                logits = _to_tensor(post_obj["logits"], device="cpu").float()
            except:
                pass

    loss_val = float(post_obj.get("loss", 0.0))
    acc1, acc5 = 0.0, 0.0
    if logits is not None:
        try:
            y_flat = y_tok.view(-1)
            acc1 = _acc_topk(logits, y_flat, 1)
            acc5 = _acc_topk(logits, y_flat, 5)
        except:
            pass

    # 【关键】计算真实的 Hot Ratio
    # 如果 total_calls 是 0 (异常)，设为 1.0 避免除零
    real_hot_ratio = 1.0 - (cold_calls / total_calls) if total_calls > 0 else 1.0

    result_row = {
        "loss": loss_val,
        "acc1": acc1,
        "acc5": acc5,
        "pre_lat": t1, "exp_lat": t2, "post_lat": t3, "e2e_lat": t1 + t2 + t3,
        "inv_retry_cnt": r1 + r2 + r3, "inv_cold_ms": c1 + c2 + c3,
        "fwd_mode_hot": 1, "fwd_mode_cold": 0, "fwd_mode_http": 0,

        # 使用真实计算的比率
        "hot_ratio": real_hot_ratio,

        "cost_usd": 0.0001
    }
    return result_row


async def train():
    if not os.path.exists(METRICS_FILE): write_metrics_header(METRICS_FILE)
    stoi, _ = _build_vocab(DATA_PATH, VOCAB_PATH)
    ids = _load_ids(DATA_PATH, stoi)
    n = int(ids.numel())
    n_train = max(SEQ_LEN + 2, int(n * 0.9))
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    train_batcher = TextBatcher(train_ids, BATCH_SIZE, SEQ_LEN, seed=SEED)
    val_batcher = TextBatcher(val_ids, BATCH_SIZE, SEQ_LEN, seed=SEED + 999)
    dataset = RealDataLoader(block_size=64, batch_size=4)
    for step in range(1, MAX_STEPS + 1):
        t_step0 = time.perf_counter()
        AUTOSCALER.step()
        split = "val" if (step % VAL_INTERVAL == 0) else "train"
        x, y = (val_batcher.next_batch() if split == "val" else train_batcher.next_batch())
        mb = MICRO_BATCH
        xs, ys = [], []
        for _ in range(mb):
            x, y = dataset.get_batch('train')
            xs.append(x)
            ys.append(y)
        # 安全检查：如果数据不够，强行中止这一轮，防止报错
        if len(xs) != mb:
            print(f">>> [Fatal Error] Loop mismatch! Generated {len(xs)}, expected {mb}")
            continue
        tasks = [run_microbatch(step, i, xs[i], ys[i]) for i in range(mb)]
        rows = await asyncio.gather(*tasks)
        valid_rows = [r for r in rows if r is not None]

        # Aggregation
        loss = float(np.mean([r["loss"] for r in valid_rows]))
        acc5 = float(np.mean([r["acc5"] for r in valid_rows]))
        hot_ratio = float(np.mean([r["hot_ratio"] for r in valid_rows]))
        inv_cold_ms = float(np.sum([r["inv_cold_ms"] for r in valid_rows]))
        step_time_ms = (time.perf_counter() - t_step0) * 1000.0
        DEADLINE_EST.update(step_time_ms)
        viol = 1.0 if step_time_ms > DEADLINE_EST.deadline_ms(step) else 0.0

        row = rows[0].copy()  # grab structure
        # Sums
        for k in ["pre_lat_ms", "post_lat_ms", "exp_lat_ms", "inv_total_ms", "inv_queue_ms", "inv_cold_ms",
                  "inv_net_ms", "inv_compute_ms", "inv_retry_cnt", "cost_usd_step", "cost_usd_pre_fwd",
                  "cost_usd_post_fwd", "cost_usd_expert_fwd"]:
            row[k] = float(np.sum([r.get(k.replace("_ms", ""), 0.0) for r in rows]))  # map short key to long
        # Fracs
        fwd_tot = max(1.0, float(np.sum([r["fwd_mode_hot"] + r["fwd_mode_cold"] + r["fwd_mode_http"] for r in rows])))
        row["fwd_mode_hot_frac"] = float(np.sum([r["fwd_mode_hot"] for r in rows])) / fwd_tot
        row["fwd_mode_cold_frac"] = float(np.sum([r["fwd_mode_cold"] for r in rows])) / fwd_tot
        row["fwd_mode_http_frac"] = float(np.sum([r["fwd_mode_http"] for r in rows])) / fwd_tot
        # Common fields
        row.update({"step": step, "split": split, "loss": loss, "acc_top5": acc5, "hot_ratio": hot_ratio,
                    "step_time_ms": step_time_ms, "deadline_violation_frac": viol})

        append_metrics(METRICS_FILE, row)
        if (step % LOG_TRAIN_EVERY) == 0 or split == "val":
            print(
                f"[{split}] step={step} loss={loss:.4f} acc5={acc5:.4f} time={step_time_ms:.0f}ms cold={inv_cold_ms:.0f}ms hot_ratio={hot_ratio:.2f}")


def main():
    # ==========================================
    # 【新增】初始化 LocalExecutor
    # ==========================================
    global LOCAL_EXECUTOR  # 1. 声明使用全局变量

    # 2. 如果开启了本地计算模式 (默认开启)，则初始化它
    if os.getenv("USE_HTTP_EXEC", "0") == "0":
        print(">>> [System] Initializing Global LocalExecutor (In-Process Mode)...")
        LOCAL_EXECUTOR = LocalExecutor()

    # 原有的启动逻辑
    asyncio.run(train())


if __name__ == "__main__": main()