import os
import random
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np


def feasible_modes() -> List[str]:
    """
    根据环境变量返回当前可用的通信模式：
    - 'hot'  : Redis 等内存级快速通道
    - 'cold' : OSS / 对象存储等高延迟通道
    - 'http' : 直接 HTTP 发送
    """
    modes: List[str] = []
    if os.getenv("REDIS_URL", ""):
        modes.append("hot")
    if os.getenv("OSS_URI_PREFIX", ""):
        modes.append("cold")
    # HTTP 始终可用，作为兜底方案
    modes.append("http")
    return modes


def estimate_objectives(
    inst: Dict[str, Any],
    mode: str,
    req: Dict[str, Any],
    deadline_ms: float,
) -> np.ndarray:
    """
    对某个 (实例, 通信模式) 组合估计 4 维目标:
    - 0: bw_time   : 总耗时 (ms)
    - 1: comm_ms   : 通信时间 (ms)
    - 2: cost      : 费用 (cent)
    - 3: stall     : 超过 deadline 的惩罚时间 (ms)
    """
    bw_table = {
        "hot": float(os.getenv("HOT_BW_MBPS", "800")),    # Redis 假设带宽高
        "cold": float(os.getenv("COLD_BW_MBPS", "200")),  # OSS 假设带宽较低
        "http": float(os.getenv("HTTP_BW_MBPS", "100")),
    }
    rtt_table = {
        "hot": float(os.getenv("HOT_RTT_MS", "2")),
        "cold": float(os.getenv("COLD_RTT_MS", "15")),
        "http": float(os.getenv("HTTP_RTT_MS", "8")),
    }
    if mode not in bw_table:
        raise ValueError(f"Unknown mode: {mode}")

    bw_mbps = max(1e-3, bw_table[mode])
    rtt_ms = rtt_table[mode]

    grad_bytes = float(req.get("grad_bytes", 1.0))
    q_ms = float(inst.get("dyn", {}).get("avg_q_ms", 0.0))

    # 传输时间: bytes -> bits / (Mbps * 1e6) -> 秒 -> ms
    tx_ms = (grad_bytes * 8.0) / (bw_mbps * 1e6) * 1000.0
    comm_ms = rtt_ms + q_ms + tx_ms
    bw_time = comm_ms

    price_cents_s = float(
        inst.get("meta", {}).get("price_cents_s", req.get("price_cents_s", 0.0))
    )
    cost = price_cents_s * (bw_time / 1000.0)

    # 冷通道通常会产生额外的 PUT 费用
    if mode == "cold":
        cost += float(os.getenv("OSS_PUT_CENTS", "0.002"))

    stall = max(0.0, bw_time - float(deadline_ms or 0.0))

    return np.array([bw_time, comm_ms, cost, stall], dtype=np.float32)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    标准 Pareto 支配关系: 所有维度 <= 且至少一维 <
    """
    return np.all(a <= b) and np.any(a < b)


def _fast_non_dominated_sort(
    pop_objs: Sequence[np.ndarray],
) -> Tuple[List[List[int]], List[int]]:
    N = len(pop_objs)
    S: List[List[int]] = [[] for _ in range(N)]
    n = [0] * N
    rank = [0] * N
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if _dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif _dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        fronts.append(Q)

    if not fronts[-1]:
        fronts.pop()
    return fronts, rank


def _crowding_distance(front: List[int], objs: Sequence[np.ndarray]) -> Dict[int, float]:
    if len(front) == 0:
        return {}
    if len(front) == 1:
        return {front[0]: float("inf")}
    if len(front) == 2:
        return {front[0]: float("inf"), front[1]: float("inf")}

    m = len(objs[0])
    distance: Dict[int, float] = {i: 0.0 for i in front}

    for j in range(m):
        values = sorted(front, key=lambda i: objs[i][j])
        distance[values[0]] = float("inf")
        distance[values[-1]] = float("inf")
        min_val = objs[values[0]][j]
        max_val = objs[values[-1]][j]
        span = max(1e-9, max_val - min_val)
        for k in range(1, len(values) - 1):
            prev_idx = values[k - 1]
            next_idx = values[k + 1]
            distance[values[k]] += (objs[next_idx][j] - objs[prev_idx][j]) / span

    return distance


def nsga2_select(
    inst_list: Sequence[Dict[str, Any]],
    req: Dict[str, Any],
    deadline_ms: float,
    pop_size: int = 24,
    generations: int = 8,
    seed: Optional[int] = None,
    modes: Optional[Sequence[str]] = None,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    使用 NSGA-II 在 (实例, 通信模式) 组合上做多目标搜索（论文级实现）。

    关键修复：
    1) **硬可行性过滤**：stall > 0 的候选直接剔除，避免用“超时换便宜”造成 Full 方案延迟虚高/不稳定。
    2) 若无可行解：fallback 到最小 bw_time（最小延迟）保证系统可运行。
    3) 在可行解空间内，再用 NSGA-II 的非支配排序 + crowding 选择，并用归一化加权得到最终解。
    """
    rnd = random.Random(seed or 1234)
    modes = list(modes) if modes is not None else feasible_modes()

    if not inst_list or not modes:
        return None

    def eval_pair(inst: Dict[str, Any], mode: str) -> np.ndarray:
        return estimate_objectives(inst, mode, req, deadline_ms)

    # ------------------------------------------------------------
    # 1) Hard feasibility filter (stall == 0)
    # ------------------------------------------------------------
    feasible_pairs: List[Tuple[Dict[str, Any], str]] = []
    feasible_objs: List[np.ndarray] = []
    for inst in inst_list:
        for mode in modes:
            obj = eval_pair(inst, mode)
            if float(obj[3]) <= 0.0:  # stall
                feasible_pairs.append((inst, mode))
                feasible_objs.append(obj)

    # No feasible -> fallback to minimum latency
    if not feasible_pairs:
        best_inst = None
        best_mode = None
        best_bw = float("inf")
        for inst in inst_list:
            for mode in modes:
                bw = float(eval_pair(inst, mode)[0])
                if bw < best_bw:
                    best_bw = bw
                    best_inst, best_mode = inst, mode
        return (best_inst, best_mode) if best_inst is not None else None

    # ------------------------------------------------------------
    # 2) NSGA-II inside feasible space
    # ------------------------------------------------------------
    def rand_ind() -> int:
        return rnd.randrange(0, len(feasible_pairs))

    def eval_ind(ind: int) -> np.ndarray:
        return feasible_objs[ind]

    pop: List[int] = [rand_ind() for _ in range(min(pop_size, len(feasible_pairs)))]
    pop_objs: List[np.ndarray] = [eval_ind(i) for i in pop]

    for _ in range(max(1, generations)):
        fronts, _ = _fast_non_dominated_sort(pop_objs)

        new_pop: List[int] = []
        new_objs: List[np.ndarray] = []
        for front in fronts:
            if len(new_pop) + len(front) > pop_size:
                dist = _crowding_distance(front, pop_objs)
                sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
                remain = pop_size - len(new_pop)
                chosen = sorted_front[:remain]
            else:
                chosen = front

            for idx in chosen:
                new_pop.append(pop[idx])
                new_objs.append(pop_objs[idx])

            if len(new_pop) >= pop_size:
                break

        pop, pop_objs = new_pop, new_objs

        # Crossover + mutation
        offspring: List[int] = []
        while len(offspring) < len(pop):
            p1 = rnd.choice(pop)
            p2 = rnd.choice(pop)
            child = p1 if rnd.random() < 0.5 else p2
            if rnd.random() < 0.15:
                child = rand_ind()
            offspring.append(child)

        combined = pop + offspring
        combined_objs = [eval_ind(i) for i in combined]
        fronts, _ = _fast_non_dominated_sort(combined_objs)

        new_pop = []
        new_objs = []
        for front in fronts:
            if len(new_pop) + len(front) > pop_size:
                dist = _crowding_distance(front, combined_objs)
                sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
                remain = pop_size - len(new_pop)
                chosen = sorted_front[:remain]
            else:
                chosen = front

            for idx in chosen:
                new_pop.append(combined[idx])
                new_objs.append(combined_objs[idx])

            if len(new_pop) >= pop_size:
                break

        pop, pop_objs = new_pop, new_objs

    # ------------------------------------------------------------
    # 3) Final pick: normalize + weighted sum (latency + cost)
    # ------------------------------------------------------------
    objs = np.stack(pop_objs, axis=0)
    minv = objs.min(axis=0)
    maxv = objs.max(axis=0)
    norm = (objs - minv) / np.maximum(maxv - minv, 1e-9)

    w_latency = float(os.getenv("NSGA_W_LATENCY", "0.5"))
    w_cost = float(os.getenv("NSGA_W_COST", "0.5"))

    score = w_latency * norm[:, 0] + w_cost * norm[:, 2]
    best_idx = int(np.argmin(score))
    inst, mode = feasible_pairs[pop[best_idx]]
    return inst, mode
