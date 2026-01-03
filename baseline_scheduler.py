# baseline_scheduler.py
from typing import Any, Dict, List, Tuple
import random
from collections import defaultdict


class BaselineScheduler:
    """
    基础调度器集合：用于对比实验 (Random, Round-Robin, Static)
    """

    def __init__(self):
        self.rr_counters = defaultdict(int)
        # 静态映射缓存: key -> instance
        self.static_map = {}

    def select_random(self, instances: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """随机选择"""
        if not instances:
            return None, 0.0
        return random.choice(instances), 0.0

    def select_round_robin(self, func_name: str, instances: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """轮询选择 (Round-Robin)"""
        if not instances:
            return None, 0.0
        n = len(instances)
        # 获取当前计数
        idx = self.rr_counters[func_name] % n
        self.rr_counters[func_name] += 1
        return instances[idx], 0.0

    def select_static(self, func_name: str, logical_id: int, instances: List[Dict[str, Any]]) -> Tuple[
        Dict[str, Any], float]:
        """
        静态映射 (Static / Stateful):
        模拟传统的非 Serverless 部署，每个 Expert 固定绑定到一个实例上。
        假设该连接是长连接 (Warm)，无冷启动。
        """
        if not instances:
            return None, 0.0

        # 唯一的映射键
        key = f"{func_name}::{logical_id}"

        if key not in self.static_map:
            # 初始化时，为了负载均衡，可以用哈希或取模来分配
            # 这里简单用取模，保证每次运行分配一致
            idx = logical_id % len(instances)
            self.static_map[key] = instances[idx]

        return self.static_map[key], 0.0


# 全局单例
BASELINE_SCHED = BaselineScheduler()