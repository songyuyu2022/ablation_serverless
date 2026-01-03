# ablation_config.py
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """
    实验配置中心：同时管理消融实验 (Ablation) 和对比实验 (Baseline)。

    环境变量控制:
    1. EXPERIMENT_TYPE: "ablation" (默认) 或 "baseline"

    2. ABLATION_MODE (仅当 TYPE=ablation 时生效):
       - full:           原本的方法 (Ours)
       - no_hotcold:     消融：去掉冷热区分
       - sync_update:    消融：强制同步更新
       - heuristic_only: 消融：去掉神经网络，只用启发式
       - predictor_only: 消融：去掉启发式，只用神经网络
       - no_nsga2:       消融：去掉 NSGA-II

    3. BASELINE_MODE (仅当 TYPE=baseline 时生效):
       - random:         随机调度 (Random)
       - round_robin:    轮询调度 (Round-Robin)
       - greedy:         贪心/启发式 (Greedy Heuristic, 无预测)
       - static:         静态有状态 (Stateful Static, 无Serverless冷启动)
       - bsp:            全同步并行 (Bulk Synchronous Parallel)
       - asp:            全异步并行 (Asynchronous Parallel)
       - ssp:            陈旧同步并行 (Stale Synchronous Parallel)
    """

    type: str = "ablation"  # "ablation" or "baseline"
    ablation_mode: str = "full"
    baseline_mode: str = "random"

    # ==========================================
    # 辅助属性：判断当前具体跑什么逻辑
    # ==========================================

    @property
    def is_ablation(self) -> bool:
        return self.type == "ablation"

    @property
    def is_baseline(self) -> bool:
        return self.type == "baseline"

    # --- 消融实验的开关 ---
    @property
    def disable_hotcold(self) -> bool:
        return self.is_ablation and self.ablation_mode == "no_hotcold"

    @property
    def force_sync_update(self) -> bool:
        # 消融的 sync_update 和基线的 BSP 都意味着“强制同步”
        if self.is_baseline and self.baseline_mode == "bsp":
            return True
        return self.is_ablation and self.ablation_mode == "sync_update"

    @property
    def heuristic_only(self) -> bool:
        # 基线的 Greedy 等同于仅使用启发式
        if self.is_baseline and self.baseline_mode == "greedy":
            return True
        return self.is_ablation and self.ablation_mode == "heuristic_only"

    @property
    def predictor_only(self) -> bool:
        return self.is_ablation and self.ablation_mode == "predictor_only"

    @property
    def disable_nsga2(self) -> bool:
        # 对比实验通常不需要跑复杂的 NSGA-II，除非是 Ours
        if self.is_baseline:
            return True
        return self.is_ablation and self.ablation_mode == "no_nsga2"

    # --- 对比实验的开关 ---
    @property
    def use_random_sched(self) -> bool:
        # ASP 通常配合随机或简单的负载均衡
        return self.is_baseline and (self.baseline_mode == "random" or self.baseline_mode == "asp")

    @property
    def use_round_robin_sched(self) -> bool:
        return self.is_baseline and self.baseline_mode == "round_robin"

    @property
    def is_static_compute(self) -> bool:
        # 静态图/有状态服务：无冷启动，固定映射
        return self.is_baseline and self.baseline_mode == "static"

    @property
    def is_ssp(self) -> bool:
        return self.is_baseline and self.baseline_mode == "ssp"

    @property
    def is_asp(self) -> bool:
        return self.is_baseline and self.baseline_mode == "asp"


def load_ablation_from_env() -> ExperimentConfig:
    exp_type = (os.getenv("EXPERIMENT_TYPE", "ablation") or "ablation").strip().lower()
    abl_mode = (os.getenv("ABLATION_MODE", "full") or "full").strip().lower()
    base_mode = (os.getenv("BASELINE_MODE", "random") or "random").strip().lower()

    # 简单的校验
    valid_types = {"ablation", "baseline"}
    if exp_type not in valid_types:
        # 默认回退，防止报错
        exp_type = "ablation"

    return ExperimentConfig(
        type=exp_type,
        ablation_mode=abl_mode,
        baseline_mode=base_mode
    )