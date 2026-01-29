# generate_instances.py (Pro Version - 基于 Azure Trace 统计规律)
import json
import random
import numpy as np  # 需要 numpy 来生成高级分布

# ==========================================
# 配置区域
# ==========================================
NUM_EXPERTS = 8
NUM_PRE_INSTANCES = 8
NUM_POST_INSTANCES = 8
NUM_EXPERT_REPLICAS = 4

# ==========================================
# 统计规律参数 (基于 Azure Functions Trace 2019)
# ==========================================
# 1. 冷启动: Log-Normal 分布 (长尾效应)
# 均值 mu=5.5 (约244ms), sigma=0.8 -> 范围主要在 100ms ~ 2000ms，偶见 5s+
COLD_START_MU = 5.5
COLD_START_SIGMA = 0.8

# 2. 性能波动: Normal 分布 (多租户干扰)
# 均值 1.0 (基准), 标准差 0.1 (10% 波动)
PERF_MU = 1.0
PERF_SIGMA = 0.1

# 3. 网络延迟: Gamma 分布 (大部分很快，少数慢)
# shape=2.0, scale=2.0 -> 均值 4ms, 长尾可达 15ms+
NET_LATENCY_SHAPE = 2.0
NET_LATENCY_SCALE = 2.0


def get_real_cold_start(is_gpu=False):
    # 基础冷启动
    val = np.random.lognormal(COLD_START_MU, COLD_START_SIGMA)
    # 限制最小值和最大值，防止过于离谱
    val = max(50.0, min(val, 15000.0))
    if is_gpu:
        # GPU 冷启动通常比 CPU 慢 2-3 倍 (加载 CUDA context)
        val *= random.uniform(2.0, 3.5)
    return round(val, 1)


def get_real_perf(base=1.0):
    # 性能因子：值越大代表性能越强 (处理越快)
    # 引入 10% 的正态波动
    noise = np.random.normal(0, PERF_SIGMA)
    val = base + noise
    return round(max(0.1, val), 2)


def get_real_net_latency():
    # 网络延迟
    val = np.random.gamma(NET_LATENCY_SHAPE, NET_LATENCY_SCALE)
    return round(max(0.5, val), 1)


def generate():
    instances = []
    func_map = {}

    # 1. 生成 Pre 实例 (CPU)
    pre_ids = []
    for i in range(NUM_PRE_INSTANCES):
        iid = f"inst_pre_{i}"
        pre_ids.append(iid)
        instances.append({
            "id": iid,
            "region": "local",
            "cpu_cores": 2,
            "memory_mb": 4096,
            "meta": {
                "performance": get_real_perf(base=1.0),
                "net_latency_ms": get_real_net_latency(),
                "price_cents_s": 0.002,
                "cold_start_ms": get_real_cold_start(is_gpu=False)  # ✅ 真实分布
            }
        })
    func_map["moe.pre_fwd"] = pre_ids

    # 2. 生成 Post 实例 (CPU)
    post_ids = []
    for i in range(NUM_POST_INSTANCES):
        iid = f"inst_post_{i}"
        post_ids.append(iid)
        instances.append({
            "id": iid,
            "region": "local",
            "cpu_cores": 2,
            "memory_mb": 4096,
            "meta": {
                "performance": get_real_perf(base=1.0),
                "net_latency_ms": get_real_net_latency(),
                "price_cents_s": 0.002,
                "cold_start_ms": get_real_cold_start(is_gpu=False)  # ✅ 真实分布
            }
        })
    func_map["moe.post_fwd"] = post_ids

    # 3. 生成 Expert 实例 (GPU)
    for e in range(NUM_EXPERTS):
        exp_ids = []
        for r in range(NUM_EXPERT_REPLICAS):
            iid = f"inst_exp{e}_rep{r}"
            exp_ids.append(iid)

            # 模拟异构性：设定不同的基准性能
            # 副本0: 高性能 (A100类)
            # 副本1: 中等 (T4类)
            # 副本2: 慢速 (旧卡)
            # 副本3: 极慢 (CPU fallback 或 拥塞卡)
            if r == 0:
                base_perf = 1.5
            elif r == 1:
                base_perf = 1.0
            elif r == 2:
                base_perf = 0.7
            else:
                base_perf = 0.4

            perf = get_real_perf(base=base_perf)
            price = 0.01 * perf

            instances.append({
                "id": iid,
                "region": "local",
                "cpu_cores": 4,
                "memory_mb": 8192,
                "meta": {
                    "device": "gpu",
                    "performance": perf,
                    "net_latency_ms": get_real_net_latency(),
                    "price_cents_s": round(price, 4),
                    "cold_start_ms": get_real_cold_start(is_gpu=True)  # ✅ GPU启动更慢
                }
            })
        func_map[f"moe.expert_fwd:{e}"] = exp_ids

    # 4. 写入文件
    with open("instances.json", "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2)

    with open("func_map.json", "w", encoding="utf-8") as f:
        json.dump(func_map, f, indent=2)

    print(f"Generated {len(instances)} instances with REALISTIC distributions.")
    print(f"Pre/Post pool size: {NUM_PRE_INSTANCES}")
    print(f"Expert Count: {NUM_EXPERTS} (Replicas per expert: {NUM_EXPERT_REPLICAS})")


if __name__ == "__main__":
    generate()