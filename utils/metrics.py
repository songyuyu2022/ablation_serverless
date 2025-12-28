# utils/metrics_ablation.py
from __future__ import annotations

"""
Drop-in MetricsLogger/StepMetrics with two extra paper-friendly fields:
- ablation_mode
- grad_selector

This file matches the StepMetrics fields used by controller_ablation_nsga2_metrics.py
so that your metrics.csv gains stable columns for ablation bookkeeping.
"""

import csv
import os
from dataclasses import dataclass, asdict, fields
from typing import List


@dataclass
class StepMetrics:
    # identifiers
    epoch: int
    step: int
    step_in_epoch: int
    phase: str

    # quality
    loss: float
    acc_top1: float
    acc_top5: float

    # workload
    batch_size: int
    seq_len: int
    tokens: int

    # step timing
    step_time_ms: float
    pre_fwd_ms: float
    post_fwd_ms: float
    expert_comm_ms: float

    # backward timing
    bwd_total_ms: float
    pre_bwd_ms: float
    post_bwd_ms: float

    # backward invoke breakdown (pre/post bwd)
    bwd_inv_total_ms: float
    bwd_inv_queue_ms: float
    bwd_inv_cold_ms: float
    bwd_inv_net_ms: float
    bwd_inv_compute_ms: float
    bwd_inv_retry_cnt: int

    # throughput
    samples_per_s: float
    tokens_per_s: float

    # grad payload
    grad_bytes: float
    grad_total: int

    # grad mode fractions
    grad_mode_hot_frac: float
    grad_mode_cold_frac: float
    grad_mode_http_frac: float
    grad_mode_local_frac: float
    grad_mode_fallback_frac: float

    # grad apply invoke breakdown
    grad_inv_total_ms: float
    grad_inv_queue_ms: float
    grad_inv_cold_ms: float
    grad_inv_net_ms: float
    grad_inv_compute_ms: float
    grad_inv_retry_cnt: int

    # nsga2 status
    grad_nsga2_feasible: int
    grad_fallback_cnt: int

    # grad per-mode latency + bytes
    grad_lat_hot_ms: float
    grad_lat_cold_ms: float
    grad_lat_http_ms: float
    grad_bytes_hot: int
    grad_bytes_cold: int
    grad_bytes_http: int

    # routing / hotcold
    dispatch_count: int
    expert_inst_cnt: int
    hot_ratio: float
    active_expert_cnt: int
    active_hot_ratio: float
    hot_flip_cnt: int
    hot_set_size: int
    hot_set_jaccard: float
    expert_load_entropy: float

    # cold accumulation stats
    cold_total_cnt: int
    cold_skipped_cnt: int
    cold_updated_cnt: int
    cold_skip_ratio: float
    cold_apply_steps_avg: float
    cold_grad_scale_avg: float
    cold_pending_steps_avg: float
    cold_update_hit_cnt: int

    # fwd mode fractions
    fwd_mode_hot_frac: float
    fwd_mode_cold_frac: float
    fwd_mode_local_frac: float
    fwd_mode_hot_frac_tok: float
    fwd_mode_cold_frac_tok: float
    fwd_mode_local_frac_tok: float

    # capacity / overflow
    capacity: int
    overflow_total_assignments: int
    overflow_dropped_assignments: int
    overflow_drop_ratio: float

    # invocation latency breakdown (expert fwd invoke)
    inv_total_ms: float
    inv_queue_ms: float
    inv_cold_ms: float
    inv_net_ms: float
    inv_compute_ms: float
    inv_retry_cnt: int

    # deadline and cost
    deadline_ms: float
    deadline_miss: int
    deadline_slack_ms: float
    cost_usd_pre_fwd: float
    cost_usd_post_fwd: float
    cost_usd_expert_fwd: float
    cost_usd_pre_bwd: float
    cost_usd_post_bwd: float
    cost_usd_grad_apply: float
    cost_usd_step: float

    # NEW tags
    ablation_mode: str = "full"
    grad_selector: str = "nsga2"


class MetricsLogger:
    def __init__(self, path: str = "metrics.csv", tail_window: int = 50):
        self.path = path
        self.tail_window = tail_window
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
