# utils/metrics.py
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, asdict, fields
from typing import List


@dataclass
class StepMetrics:
    # --- 1. 基础标识 (必填) ---
    epoch: int
    step: int
    phase: str  # train / val

    # --- 2. 训练质量 (Fig 1: 收敛曲线) ---
    loss: float = 0.0
    acc_top5: float = 0.0

    # --- 3. 性能分析 (Fig 2 / Fig 3: 步时长与稳定性) ---
    step_time_ms: float = 0.0
    # 【新增】滚动长尾/稳定性指标（最近窗口）
    step_time_p95_ms: float = 0.0
    step_time_p99_ms: float = 0.0
    step_time_cv: float = 0.0

    # global（从开始到当前 step 的累计）
    step_time_global_p95_ms: float = 0.0
    step_time_global_p99_ms: float = 0.0
    step_time_global_cv: float = 0.0

    # 【Fig Y】预测器准确性
    predictor_r2: float = 0.0
    predictor_mae: float = 0.0

    # 【Fig W-a】冷启动统计
    inv_cold_cnt: int = 0
    inv_cold_ms: float = 0.0

    # --- 4. 经济成本 (Fig W-c: 成本细分) ---
    cost_usd_step: float = 0.0
    cost_usd_pre_fwd: float = 0.0
    cost_usd_post_fwd: float = 0.0
    cost_usd_expert_fwd: float = 0.0

    # --- 5. 辅助与实验元数据 ---
    hot_ratio: float = 0.0
    ablation_mode: str = "full"
    grad_selector: str = "nsga2"

    inv_net_ms: float = 0.0
    inv_compute_ms: float = 0.0
    inv_queue_ms: float = 0.0
    deadline_ms: float = 0.0
    deadline_violation_frac: float = 0.0


class MetricsLogger:
    def __init__(self, path: str = "metrics.csv"):
        self.path = path
        self._header: List[str] = [f.name for f in fields(StepMetrics)]
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._header)
            w.writeheader()

    def log(self, m: StepMetrics) -> None:
        row = asdict(m)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._header)
            w.writerow({k: row.get(k, "") for k in self._header})