# scheduler_hybrid.py
from typing import Any, Dict, List, Tuple
import numpy as np
import os

from scheduler_heuristic import HeuristicScheduler, DEFAULT_HEURISTIC_SCHED
from scheduler_nn import NNScheduler, NN_SCHED
from utils.logger import log
from ablation_config import load_ablation_from_env


class HybridScheduler:
    """
    Hybrid = Heuristic (Base) + Online NN (Correction)

    关键修复：
    - 默认 HYBRID_NN_WEIGHT 从 0.5 调低到 0.2：避免在线预测器“还没学好就把曲线搞抖”
    - 加 Guardrail：若 NN 融合后最优值显著变差（> 1.2x），本轮退回纯 heuristic
    """

    def __init__(
        self,
        base_sched: HeuristicScheduler | None = None,
        nn_sched: NNScheduler | None = None,
    ) -> None:
        self.base_sched = base_sched or DEFAULT_HEURISTIC_SCHED
        self.nn_sched = nn_sched or NN_SCHED

        # safer default for paper runs
        self.nn_weight = float(os.getenv("HYBRID_NN_WEIGHT", "0.2"))

        self._abl = load_ablation_from_env()
        # Ablation toggles
        self.disable_nn = self._abl.heuristic_only
        self.disable_heuristic = self._abl.predictor_only
        if self.disable_nn:
            self.nn_weight = 0.0

    def select_instances(
        self,
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
        top_k: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:

        if not instances:
            raise RuntimeError("HybridScheduler: instances empty")

        # 1. 获取基础分 (稳定、托底)
        if self.disable_heuristic:
            base_scores = [0.0] * len(instances)
        else:
            base_scores = self.base_sched.get_scores(func_type, logical_id, instances, req)

        # 2. 获取 AI 预测分 (动态、学习)
        if self.disable_nn:
            nn_scores = [0.0] * len(instances)
        else:
            nn_scores = self.nn_sched.get_scores(func_type, logical_id, instances, req)

        # 3. 融合: Final = Base + w * NN
        final_scores = []
        for b, n in zip(base_scores, nn_scores):
            final_scores.append(b + self.nn_weight * n)

        # 3.5 Guardrail: if NN hurts too much, fall back to heuristic
        if not self.disable_heuristic and not self.disable_nn:
            best_final = float(min(final_scores))
            best_base = float(min(base_scores))
            # if NN makes the best choice much worse, ignore NN this round
            if best_final > best_base * 1.2:
                final_scores = list(base_scores)

        # 4. 排序（分数越小越好）
        order = np.argsort(final_scores)
        chosen_idx = order[:top_k]

        return [instances[i] for i in chosen_idx], [final_scores[i] for i in chosen_idx]

    def select_instance(
        self,
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        chosen, scores = self.select_instances(func_type, logical_id, instances, req, top_k=1)
        return chosen[0], scores[0]

    def update_stats(
        self,
        func_type: str,
        logical_id: int,
        inst: Dict[str, Any],
        req: Dict[str, Any],
        latency_ms: float,
    ):
        """在线学习闭环：将真实 Latency 反馈给 NN"""
        if self.disable_nn:
            return
        try:
            self.nn_sched.update(func_type, logical_id, inst, req, latency_ms)
        except Exception as e:
            log("hybrid", f"update_stats failed: {e}")


HYBRID_SCHED = HybridScheduler()
