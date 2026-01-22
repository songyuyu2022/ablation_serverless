import json
import os
import random
import copy
import numpy as np

# ================= 配置区域 =================

# 关键改动1：稀缺性控制
# 我们不让每种类型都生成一样多，而是让高端卡更少
REPLICAS_MAP = {
    "A100": 4,  # 只有 4 个 A100 (稀缺，容易排队)
    "V100": 8,  # 8 个 V100
    "T4": 20,  # 20 个 T4 (充裕，不用排队)
    "CPU": 32  # 32 个 CPU
}

# 关键改动2：引入 MIG 切分 (A100_MIG)
# 模拟 "计算快但显存小" 的情况
GPU_TIERS = [
    # --- 顶级全卡 (完美，但极贵、极少) ---
    {
        "name": "NVIDIA_A100_80G",
        "family": "A100",
        "prob": 0.05,  # 只有 5% 的概率抽到完整卡
        "perf_base": 5.0,
        "mem_base": 81920,  # 80GB
        "sys_mem": 128 * 1024,
        "cpu_base": 24,
        "price": 0.40,
        "desc": "A100 Full"
    },
    # --- A100 MIG 切分 (陷阱：算力强，显存小) ---
    {
        "name": "NVIDIA_A100_MIG_1g.5gb",
        "family": "A100",
        "prob": 0.15,  # 15% 概率是这种切片
        "perf_base": 4.5,  # 算力依然是 A100 级别的强 (架构优势)
        "mem_base": 5120,  # 【坑】只有 5GB 显存！容易 OOM
        "sys_mem": 16 * 1024,
        "cpu_base": 2,  # 配套 CPU 也很弱
        "price": 0.08,  # 便宜
        "desc": "A100 MIG Slice (High Perf, Low Mem)"
    },
    # --- 主流卡 (V100) ---
    {
        "name": "NVIDIA_V100_32G",
        "family": "V100",
        "prob": 0.30,
        "perf_base": 2.5,
        "mem_base": 32768,
        "sys_mem": 64 * 1024,
        "cpu_base": 8,
        "price": 0.18,
        "desc": "V100 Mainstream"
    },
    # --- 低端卡 (T4) ---
    {
        "name": "NVIDIA_T4_16G",
        "family": "T4",
        "prob": 0.50,  # 50% 都是这种低端卡
        "perf_base": 1.0,
        "mem_base": 16384,
        "sys_mem": 32 * 1024,
        "cpu_base": 4,
        "price": 0.06,
        "desc": "T4 Entry"
    }
]

CPU_TIERS = [
    {"name": "Intel_Platinum", "family": "CPU", "prob": 0.2, "perf_base": 1.2, "sys_mem": 32 * 1024, "cpu_base": 8,
     "price": 0.002, "desc": "Fast CPU"},
    {"name": "Intel_Xeon", "family": "CPU", "prob": 0.8, "perf_base": 0.6, "sys_mem": 16 * 1024, "cpu_base": 4,
     "price": 0.001, "desc": "Slow CPU"},
]

PERF_JITTER = 0.10

# ================= 基础模板 =================
# 这里的 "type" 决定了它去哪个池子抽奖
RAW_TEMPLATES = [
    {"base_id": "fn_pre_gpu", "func_name": "moe.pre_fwd", "type": "gpu", "start_port": 10000,
     "base_meta": {"device": "cuda", "cold_start_ms": 500.0}},
    {"base_id": "fn_pre_cpu", "func_name": "moe.pre_fwd", "type": "cpu", "start_port": 11000,
     "base_meta": {"device": "cpu", "cold_start_ms": 200.0}},
    {"base_id": "fn_post_gpu", "func_name": "moe.post_fwd", "type": "gpu", "start_port": 12000,
     "base_meta": {"device": "cuda", "cold_start_ms": 500.0}},
    {"base_id": "fn_post_cpu", "func_name": "moe.post_fwd", "type": "cpu", "start_port": 13000,
     "base_meta": {"device": "cpu", "cold_start_ms": 200.0}},
]

NUM_EXPERTS = 4
for i in range(NUM_EXPERTS):
    # Expert 既可能是 GPU 也可能是 CPU
    RAW_TEMPLATES.append(
        {"base_id": f"fn_exp{i}_gpu", "func_name": f"moe.expert_fwd:{i}", "type": "gpu", "start_port": 20000 + i * 1000,
         "base_meta": {"device": "cuda", "cold_start_ms": 800.0, "gpu_request": 1}})
    RAW_TEMPLATES.append(
        {"base_id": f"fn_exp{i}_cpu", "func_name": f"moe.expert_fwd:{i}", "type": "cpu", "start_port": 30000 + i * 1000,
         "base_meta": {"device": "cpu", "cold_start_ms": 250.0}})


# ================= 生成逻辑 =================
def pick_tier(tiers):
    r = random.random()
    cum = 0.0
    for t in tiers:
        cum += t["prob"]
        if r <= cum: return t
    return tiers[-1]


instances = []
func_map = {}

print(f"Generating instances with MIG Slicing & Scarcity constraints...")
random.seed(42)

for t in RAW_TEMPLATES:
    func_name = t["func_name"]
    if func_name not in func_map: func_map[func_name] = []

    tier_pool = GPU_TIERS if t["type"] == "gpu" else CPU_TIERS

    # 动态决定生成多少个副本
    # 我们先预生成一批，然后根据稀缺性过滤
    # 这里简化处理：直接生成一大批，然后根据概率分布，自然就会出现 A100 少 T4 多的情况
    # 因为 GPU_TIERS 里的 prob 已经定义了 A100 只有 0.05

    # 为了保证总数足够，我们生成 30 个，依靠概率来控制分布
    TOTAL_GEN = 30

    for r in range(1, TOTAL_GEN + 1):
        inst_id = f"{t['base_id']}_{r}"
        port = t['start_port'] + r

        # 1. 抽取硬件档次
        tier = pick_tier(tier_pool)

        # 2. 注入 Jitter
        jitter = random.uniform(1.0 - PERF_JITTER, 1.0 + PERF_JITTER)
        final_perf = round(tier["perf_base"] * jitter, 3)

        inst = {
            "id": inst_id,
            "url": f"http://127.0.0.1:{port}",
            "runtime": "python3.11" if t["type"] == "gpu" else "python3.10",
            "memory_mb": tier["sys_mem"],
            "cpu_cores": tier["cpu_base"],
            "region": f"local-{t['type']}-pool",
            "meta": copy.deepcopy(t["base_meta"])
        }

        # 3. 填充 Meta
        inst["meta"].update({
            "desc": f"{tier['desc']} #{r}",
            "tier": tier["name"],
            "performance": final_perf,
            "price_cents_s": tier["price"],
            "gpu_mem_mb": tier.get("mem_base", 0)  # 显存
        })

        instances.append(inst)
        func_map[func_name].append(inst_id)

with open("instances.json", "w", encoding="utf-8") as f:
    json.dump(instances, f, indent=2, ensure_ascii=False)
with open("func_map.json", "w", encoding="utf-8") as f:
    json.dump(func_map, f, indent=2, ensure_ascii=False)

print(f"✅ Generated {len(instances)} instances.")
print("=== Sample Distribution ===")
tiers = [i['meta']['tier'] for i in instances if 'gpu' in i['id']]
from collections import Counter

print(Counter(tiers))