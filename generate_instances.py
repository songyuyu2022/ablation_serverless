# generate_instances.py (Enhanced - heterogeneity + comm/compute variability)
# 目标：
# 1) 让通信(net)不再“几乎恒定”：引入多 region / 混合分布 / 更合理的 RTT 量级（本地~5ms，跨区~50-150ms）
# 2) 让计算(compute)更有差异：扩大 perf 抖动、引入更明显的实例异构（不同 GPU 档位 + 轻度相关噪声）
# 3) 与 controller 兼容：同时写入 rtt_ms 和 net_latency_ms（controller 优先读 rtt_ms，否则读 net_latency_ms）
#
# 用法：
#   python generate_instances.py
# 产物：
#   instances.json / func_map.json

import json
import os
import random
from typing import Dict, Any

import numpy as np

# ==========================================
# 可通过环境变量覆盖的配置
# ==========================================
NUM_EXPERTS = int(os.getenv("NUM_EXPERTS", "8"))
NUM_PRE_INSTANCES = int(os.getenv("NUM_PRE_INSTANCES", "8"))
NUM_POST_INSTANCES = int(os.getenv("NUM_POST_INSTANCES", "8"))
NUM_EXPERT_REPLICAS = int(os.getenv("NUM_EXPERT_REPLICAS", "4"))

# ==========================================
# 冷启动分布（Lognormal，长尾）
# ==========================================
COLD_START_MU = float(os.getenv("COLD_START_MU", "5.5"))
COLD_START_SIGMA = float(os.getenv("COLD_START_SIGMA", "0.8"))
COLD_START_MIN_MS = float(os.getenv("COLD_START_MIN_MS", "50.0"))
COLD_START_MAX_MS = float(os.getenv("COLD_START_MAX_MS", "15000.0"))

# GPU 冷启动倍率
GPU_COLD_MUL_MIN = float(os.getenv("GPU_COLD_MUL_MIN", "2.0"))
GPU_COLD_MUL_MAX = float(os.getenv("GPU_COLD_MUL_MAX", "3.5"))

# ==========================================
# 性能（performance）分布：扩大抖动
# ==========================================
PERF_SIGMA = float(os.getenv("PERF_SIGMA", "0.20"))
PERF_MIN = float(os.getenv("PERF_MIN", "0.10"))

# ==========================================
# Region Profiles：让通信时延可区分
# ==========================================
REGION_PROFILES = {
    "local":  {"region": "local", "rtt_q50": 5.0,   "rtt_q90": 12.0,  "bw_mbps": 10000, "price_mul": 1.00},
    "near":   {"region": "near",  "rtt_q50": 35.0,  "rtt_q90": 80.0,  "bw_mbps": 2000,  "price_mul": 1.05},
    "far":    {"region": "far",   "rtt_q50": 90.0,  "rtt_q90": 180.0, "bw_mbps": 500,   "price_mul": 1.10},
}


def _z(p: float) -> float:
    # z0.9 ≈ 1.28155
    if abs(p - 0.90) < 1e-6:
        return 1.281551565545
    return 1.0


def _sample_lognormal_from_q(q50: float, q90: float, rng: random.Random) -> float:
    import math
    q50 = max(float(q50), 1e-12)
    q90 = max(float(q90), q50 * 1.000001)
    mu = math.log(q50)
    sigma = (math.log(q90) - mu) / max(_z(0.90), 1e-9)
    return float(math.exp(rng.gauss(mu, sigma)))


def get_real_cold_start(rng: random.Random, *, is_gpu: bool = False) -> float:
    val = float(np.random.lognormal(COLD_START_MU, COLD_START_SIGMA))
    val = max(COLD_START_MIN_MS, min(val, COLD_START_MAX_MS))
    if is_gpu:
        val *= rng.uniform(GPU_COLD_MUL_MIN, GPU_COLD_MUL_MAX)
    return round(val, 1)


def get_real_perf(*, base: float = 1.0) -> float:
    noise = float(np.random.normal(0.0, PERF_SIGMA))
    val = base + noise
    return round(max(PERF_MIN, val), 2)


def get_real_rtt_ms(rng: random.Random, profile: Dict[str, Any]) -> float:
    q50, q90 = float(profile["rtt_q50"]), float(profile["rtt_q90"])
    x = _sample_lognormal_from_q(q50, q90, rng)
    return round(max(0.5, x), 1)


def choose_region_for_pool(rng: random.Random, pool: str) -> Dict[str, Any]:
    if pool in ("pre", "post"):
        r = rng.random()
        if r < 0.80:
            return REGION_PROFILES["local"]
        if r < 0.95:
            return REGION_PROFILES["near"]
        return REGION_PROFILES["far"]
    else:
        r = rng.random()
        if r < 0.40:
            return REGION_PROFILES["local"]
        if r < 0.70:
            return REGION_PROFILES["near"]
        return REGION_PROFILES["far"]


def instance_price_cents_s(base_cents_s: float, *, perf: float, profile: Dict[str, Any]) -> float:
    mul = float(profile.get("price_mul", 1.0))
    return round(base_cents_s * mul * max(0.2, perf), 4)


def generate(seed: int = 123):
    rng = random.Random(seed)
    np.random.seed(seed)

    instances = []
    func_map: Dict[str, Any] = {}

    # 1) Pre
    pre_ids = []
    for i in range(NUM_PRE_INSTANCES):
        iid = f"inst_pre_{i}"
        pre_ids.append(iid)

        prof = choose_region_for_pool(rng, "pre")
        perf = get_real_perf(base=1.0)
        rtt = get_real_rtt_ms(rng, prof)

        instances.append({
            "id": iid,
            "region": prof["region"],
            "cpu_cores": 2,
            "memory_mb": 4096,
            "max_concurrency": int(os.getenv("PRE_MAX_CONCURRENCY", "8")),
            "meta": {
                "performance": perf,
                "rtt_ms": rtt,
                "net_latency_ms": rtt,
                "bandwidth_mbps": prof["bw_mbps"],
                "price_cents_s": instance_price_cents_s(0.002, perf=perf, profile=prof),
                "cold_start_ms": get_real_cold_start(rng, is_gpu=False),
            }
        })
    func_map["moe.pre_fwd"] = pre_ids

    # 2) Post
    post_ids = []
    for i in range(NUM_POST_INSTANCES):
        iid = f"inst_post_{i}"
        post_ids.append(iid)

        prof = choose_region_for_pool(rng, "post")
        perf = get_real_perf(base=1.0)
        rtt = get_real_rtt_ms(rng, prof)

        instances.append({
            "id": iid,
            "region": prof["region"],
            "cpu_cores": 2,
            "memory_mb": 4096,
            "max_concurrency": int(os.getenv("POST_MAX_CONCURRENCY", "8")),
            "meta": {
                "performance": perf,
                "rtt_ms": rtt,
                "net_latency_ms": rtt,
                "bandwidth_mbps": prof["bw_mbps"],
                "price_cents_s": instance_price_cents_s(0.002, perf=perf, profile=prof),
                "cold_start_ms": get_real_cold_start(rng, is_gpu=False),
            }
        })
    func_map["moe.post_fwd"] = post_ids

    # 3) Experts
    for e in range(NUM_EXPERTS):
        exp_ids = []
        for r in range(NUM_EXPERT_REPLICAS):
            iid = f"inst_exp{e}_rep{r}"
            exp_ids.append(iid)

            if r == 0:
                base_perf = 1.6
            elif r == 1:
                base_perf = 1.1
            elif r == 2:
                base_perf = 0.7
            else:
                base_perf = 0.45

            prof = choose_region_for_pool(rng, "expert")
            perf = get_real_perf(base=base_perf)
            rtt = get_real_rtt_ms(rng, prof)
            price = instance_price_cents_s(0.010, perf=perf, profile=prof)

            instances.append({
                "id": iid,
                "region": prof["region"],
                "cpu_cores": 4,
                "memory_mb": 8192,
                "max_concurrency": int(os.getenv("EXP_MAX_CONCURRENCY", "2")),
                "meta": {
                    "device": "gpu",
                    "performance": perf,
                    "rtt_ms": rtt,
                    "net_latency_ms": rtt,
                    "bandwidth_mbps": prof["bw_mbps"],
                    "price_cents_s": price,
                    "cold_start_ms": get_real_cold_start(rng, is_gpu=True),
                }
            })
        func_map[f"moe.expert_fwd:{e}"] = exp_ids

    with open("instances.json", "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2)

    with open("func_map.json", "w", encoding="utf-8") as f:
        json.dump(func_map, f, indent=2)

    print(f"Generated {len(instances)} instances.")
    print(f"Pre/Post pool size: {NUM_PRE_INSTANCES}/{NUM_POST_INSTANCES}")
    print(f"Expert Count: {NUM_EXPERTS} (Replicas per expert: {NUM_EXPERT_REPLICAS})")
    print("Region mix enabled: local/near/far")
    print("meta includes both rtt_ms and net_latency_ms for controller compatibility.")


if __name__ == "__main__":
    SEED = int(os.getenv("GEN_INST_SEED", "123"))
    generate(SEED)
