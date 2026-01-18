# tools/fit_alibaba_gpu_stats.py
# -*- coding: utf-8 -*-

"""
Alibaba 2025 GPU trace fitter (your disaggregated_DLRM_trace.csv).

✅ No CLI needed. Just run:
   python tools/fit_alibaba_gpu_stats.py
(or right-click run in PyCharm)

This script supports:
- Your actual schema:
  instance_sn, role, app_name, ... , creation_time, scheduled_time, deletion_time
  - app := app_name
  - func := role
  - start := scheduled_time (fallback to creation_time)
  - end := deletion_time
  - duration := end - start
  - only keep GPU jobs by default: gpu_request > 0

Output:
- ./tools/calib/alibaba2025_gpu_profile.json

What it learns:
- duration distribution (global + topK app|role pairs)
- concurrency distribution via sweep-line on (start,end) intervals
- recommended GPU capacity (cap_p95/cap_p99 of concurrency)
- basic resource request summaries (gpu_request/cpu/memory/rdma) for realism calibration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# =========================
# USER SETTINGS (edit if needed)
# =========================
TOOLS_DIR = Path(__file__).resolve().parent
GPU_CSV = TOOLS_DIR / "disaggregated_DLRM_trace.csv"
OUT_JSON = TOOLS_DIR / "calib" / "alibaba2025_gpu_profile.json"

# Keep top-K (app|role) pairs; rest aggregated into "__OTHER__"
TOPK_PAIRS = 2000

# Filter only GPU jobs (recommended for your "GPU realism" goal)
ONLY_GPU_REQUEST_GT0 = True

# If your timestamps are in milliseconds, set TIME_UNIT="ms"
TIME_UNIT = "auto"  # "auto" / "s" / "ms"

# =========================
# Required columns for your file
# =========================
REQ_COLS = [
    "instance_sn", "role", "app_name",
    "cpu_request", "cpu_limit",
    "gpu_request", "gpu_limit",
    "rdma_request", "rdma_limit",
    "memory_request", "memory_limit",
    "disk_request", "disk_limit",
    "max_instance_per_node",
    "creation_time", "scheduled_time", "deletion_time",
]


def _safe_log_quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.95, 0.99)):
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return {f"q{int(q*100)}": None for q in qs}
    lx = np.log(x)
    return {f"q{int(q*100)}": float(np.exp(np.quantile(lx, q))) for q in qs}


def _stats_float(x: np.ndarray) -> Dict[str, Any]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0,
            "mean": None, "p50": None, "p90": None, "p95": None, "p99": None,
            "min": None, "max": None,
            "log_quantiles": _safe_log_quantiles(x),
        }
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "log_quantiles": _safe_log_quantiles(x),
    }


def _stats_int(x: np.ndarray) -> Dict[str, Any]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "p50": int(np.quantile(x, 0.50)),
        "p90": int(np.quantile(x, 0.90)),
        "p95": int(np.quantile(x, 0.95)),
        "p99": int(np.quantile(x, 0.99)),
        "max": int(np.max(x)),
    }


def _infer_time_scale_to_seconds(ts: np.ndarray, mode: str) -> float:
    if mode == "s":
        return 1.0
    if mode == "ms":
        return 1e-3
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        return 1.0
    mx = float(np.max(ts))
    # heuristic: if max is huge, likely ms
    return 1e-3 if mx > 1e12 else 1.0


def _concurrency_samples_from_intervals(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Sweep-line concurrency at event points (exact, no binning).
    """
    events_t = np.concatenate([starts, ends])
    events_d = np.concatenate([
        np.ones_like(starts, dtype=np.int32),
        -np.ones_like(ends, dtype=np.int32),
    ])
    order = np.argsort(events_t, kind="mergesort")
    events_d = events_d[order]
    c = 0
    out = np.empty(events_d.shape[0], dtype=np.int32)
    for i, d in enumerate(events_d):
        c += int(d)
        out[i] = c
    return out


def _pick_start_time(df: pd.DataFrame) -> pd.Series:
    """
    start_time = scheduled_time if available else creation_time
    """
    sch = pd.to_numeric(df["scheduled_time"], errors="coerce")
    cre = pd.to_numeric(df["creation_time"], errors="coerce")
    start = sch.copy()
    start[sch.isna()] = cre[sch.isna()]
    return start


def fit_gpu_profile():
    if not GPU_CSV.exists():
        raise FileNotFoundError(
            f"[GPUFit] Cannot find: {GPU_CSV}\n"
            f"Put disaggregated_DLRM_trace.csv under tools/ (same level as this script)."
        )

    df = pd.read_csv(GPU_CSV)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"[GPUFit] Missing columns: {missing}\nFound columns: {df.columns.tolist()}")

    # normalize types
    df = df.copy()
    df["app"] = df["app_name"].astype(str)
    df["func"] = df["role"].astype(str)

    # numeric fields
    for c in ["cpu_request", "cpu_limit", "gpu_request", "gpu_limit",
              "rdma_request", "rdma_limit", "memory_request", "memory_limit",
              "disk_request", "disk_limit", "max_instance_per_node",
              "creation_time", "scheduled_time", "deletion_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter GPU jobs if enabled
    if ONLY_GPU_REQUEST_GT0:
        df = df[df["gpu_request"].fillna(0) > 0].copy()

    # choose start/end
    df["start_raw"] = _pick_start_time(df)
    df["end_raw"] = df["deletion_time"]

    # drop rows without end or start
    df = df.dropna(subset=["start_raw", "end_raw"]).copy()

    # infer time unit scale
    scale = _infer_time_scale_to_seconds(df["end_raw"].to_numpy(dtype=np.float64), TIME_UNIT)
    df["start_s"] = df["start_raw"].astype(np.float64) * scale
    df["end_s"] = df["end_raw"].astype(np.float64) * scale

    # duration
    df["dur_s"] = df["end_s"] - df["start_s"]
    df = df[df["dur_s"] > 0].copy()  # remove non-positive or corrupted intervals

    # pair key = app|func
    df["pair"] = df["app"] + "|" + df["func"]
    pair_counts = df["pair"].value_counts()
    top_pairs = set(pair_counts.head(max(1, TOPK_PAIRS)).index.tolist())

    # concurrency
    starts = df["start_s"].to_numpy(dtype=np.float64)
    ends = df["end_s"].to_numpy(dtype=np.float64)
    conc = _concurrency_samples_from_intervals(starts, ends) if starts.size else np.array([], dtype=np.int32)

    # resource summaries (for realism)
    res_summary = {
        "gpu_request": _stats_float(df["gpu_request"].to_numpy(dtype=np.float64)),
        "gpu_limit": _stats_float(df["gpu_limit"].to_numpy(dtype=np.float64)),
        "cpu_request": _stats_float(df["cpu_request"].to_numpy(dtype=np.float64)),
        "cpu_limit": _stats_float(df["cpu_limit"].to_numpy(dtype=np.float64)),
        "memory_request": _stats_float(df["memory_request"].to_numpy(dtype=np.float64)),
        "memory_limit": _stats_float(df["memory_limit"].to_numpy(dtype=np.float64)),
        "rdma_request": _stats_float(df["rdma_request"].to_numpy(dtype=np.float64)),
        "rdma_limit": _stats_float(df["rdma_limit"].to_numpy(dtype=np.float64)),
    }

    profile: Dict[str, Any] = {
        "source": "Alibaba_disaggregated_DLRM_trace",
        "input_file": str(GPU_CSV),
        "only_gpu_request_gt0": bool(ONLY_GPU_REQUEST_GT0),
        "time_scale_to_seconds": float(scale),
        "topk_pairs": int(TOPK_PAIRS),

        "global": {},
        "pairs": {},
        "other_aggregate": {},
        "capacity_recommendation": {},
        "resource_summary": res_summary,

        "notes": {
            "pair_key": "app_name|role",
            "start_time_rule": "scheduled_time if present else creation_time",
            "end_time_rule": "deletion_time",
            "duration_rule": "dur_s = end_s - start_s (filtered dur_s > 0)",
            "concurrency_method": "sweep_line_on_intervals",
        }
    }

    # global duration stats
    profile["global"]["duration_s"] = _stats_float(df["dur_s"].to_numpy(dtype=np.float64))

    # global concurrency stats and capacity recommendation
    if conc.size:
        conc_stats = _stats_int(conc.astype(np.float64))
        profile["global"]["concurrency_event_samples"] = conc_stats

        cap_p95 = int(conc_stats["p95"]) if conc_stats.get("p95") is not None else 0
        cap_p99 = int(conc_stats["p99"]) if conc_stats.get("p99") is not None else 0
    else:
        profile["global"]["concurrency_event_samples"] = {}
        cap_p95, cap_p99 = 0, 0

    profile["capacity_recommendation"] = {
        "cap_p95": cap_p95,
        "cap_p99": cap_p99,
        "hint": "Use cap_p95 as default simulated GPU pool size; cap_p99 for stress-test."
    }

    # per-pair duration stats
    other_durs = []
    for pair, cnt in pair_counts.items():
        arr = df.loc[df["pair"] == pair, "dur_s"].to_numpy(dtype=np.float64)
        if pair in top_pairs:
            profile["pairs"][pair] = {
                "count": int(cnt),
                "duration_s": _stats_float(arr),
            }
        else:
            if arr.size:
                other_durs.append(arr)

    if other_durs:
        od = np.concatenate(other_durs)
        profile["other_aggregate"]["duration_s"] = _stats_float(od)

    # write output
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"[GPUFit][OK] wrote: {OUT_JSON}")
    print(f"[GPUFit] rows_used={len(df)} ; total_pairs={len(pair_counts)} ; top_saved={min(TOPK_PAIRS, len(pair_counts))}")
    if cap_p95:
        print(f"[GPUFit] recommended cap_p95={cap_p95}, cap_p99={cap_p99}")
    else:
        print("[GPUFit][WARN] concurrency stats empty (check timestamps / filtering).")


if __name__ == "__main__":
    fit_gpu_profile()
