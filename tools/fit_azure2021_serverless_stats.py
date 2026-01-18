# tools/fit_azure2021_serverless_stats.py
# -*- coding: utf-8 -*-

"""
Standalone fitter for Azure Functions Invocation Trace 2021 (statistical model).

✅ No CLI needed. Just run:
   python tools/fit_azure2021_serverless_stats.py
or run in IDE.

Expected simplified schema (header columns):
  app, func, end_timestamp, duration

This script learns:
- per (app|func) inter-arrival gap distribution (based on end_timestamp)
- per (app|func) duration distribution
- inferred cold probability: P(gap > IDLE_THRESHOLD_S)
- also global + other_aggregate stats

Output:
- ./tools/calib/azure2021_profile.json

Enhancements vs your current version:
- auto-detect input file by schema under tools/
- robust delimiter handling for .txt/.csv
- chunked reading for large files
- drop/clamp non-positive durations to avoid p50=0 artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================
TOOLS_DIR = Path(__file__).resolve().parent

# Output path
OUT_JSON = TOOLS_DIR / "calib" / "azure2021_profile.json"

# cold candidate: idle gap > threshold
IDLE_THRESHOLD_S = 300.0

# keep top-K (app|func) pairs, rest aggregated into "__OTHER__"
TOPK_PAIRS = 2000

# units
TIME_UNIT = "auto"      # "auto" / "s" / "ms" / "us"
DURATION_UNIT = "auto"  # "auto" / "s" / "ms"

# expected columns
REQ_COLS = ["app", "func", "end_timestamp", "duration"]

# Auto search input file under tools/ (you can add subfolders if you want)
SEARCH_DIRS = [
    TOOLS_DIR,
    TOOLS_DIR / "azure2021",
]

# Candidate extensions
ALLOWED_EXT = {".csv", ".txt", ".tsv"}

# For large files: read in chunks (increase if your machine has lots of RAM)
CHUNK_ROWS = 500_000

# Realism fix: drop duration <= 0 and (optionally) clamp very small durations
DROP_NONPOSITIVE_DURATION = True
CLAMP_MIN_DURATION_S = 0.001  # 1ms; set to 0 to disable clamping


# =========================
# helpers
# =========================
def _infer_ts_scale(ts_vals: np.ndarray, mode: str) -> float:
    if mode == "s":
        return 1.0
    if mode == "ms":
        return 1e-3
    if mode == "us":
        return 1e-6
    ts_vals = ts_vals[np.isfinite(ts_vals)]
    if ts_vals.size == 0:
        return 1.0
    mx = float(np.nanmax(ts_vals))
    # heuristic: 1e12 => us-ish, 1e9 => ms-ish
    if mx > 1e12:
        return 1e-6
    if mx > 1e9:
        return 1e-3
    return 1.0


def _infer_dur_scale(dur_vals: np.ndarray, mode: str) -> float:
    if mode == "s":
        return 1.0
    if mode == "ms":
        return 1e-3
    dur_vals = dur_vals[np.isfinite(dur_vals)]
    if dur_vals.size == 0:
        return 1.0
    p90 = float(np.nanquantile(dur_vals, 0.9))
    return 1e-3 if p90 > 500 else 1.0


def _safe_log_quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, Optional[float]]:
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return {f"q{int(q*100)}": None for q in qs}
    lx = np.log(x)
    return {f"q{int(q*100)}": float(np.exp(np.quantile(lx, q))) for q in qs}


def _stats(x: np.ndarray) -> Dict[str, Optional[float]]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": None, "p50": None, "p90": None, "p95": None, "p99": None,
            "log_quantiles": _safe_log_quantiles(x),
        }
    return {
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "log_quantiles": _safe_log_quantiles(x),
    }


def _try_read_head(path: Path, nrows: int = 5) -> Optional[pd.DataFrame]:
    """
    Try reading file header with a few strategies:
    - default CSV (comma)
    - tab
    - python engine with sep=None (auto)
    """
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep="\t", nrows=nrows)
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    except Exception:
        return None


def _auto_find_azure_file() -> Path:
    required = set(REQ_COLS)

    candidates: List[Path] = []
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in ALLOWED_EXT:
                continue
            candidates.append(p)

    # deterministic order
    candidates = sorted(candidates, key=lambda x: x.name.lower())

    for p in candidates:
        head = _try_read_head(p, nrows=5)
        if head is None or head.empty:
            continue
        cols = set([c.strip() for c in head.columns])
        if required.issubset(cols):
            return p

    raise FileNotFoundError(
        "[AzureFit] Cannot auto-detect Azure trace file.\n"
        f"Expected columns: {REQ_COLS}\n"
        "Searched:\n" + "\n".join(f"  - {d}" for d in SEARCH_DIRS) + "\n"
        "Place your Azure trace (csv/txt/tsv) under tools/ or tools/azure2021/."
    )


def _iter_chunks(path: Path) -> Tuple[pd.io.parsers.TextFileReader, str]:
    """
    Return a chunk iterator + a human-readable 'dialect' description.
    """
    # Attempt 1: comma
    try:
        it = pd.read_csv(path, chunksize=CHUNK_ROWS)
        return it, "comma"
    except Exception:
        pass
    # Attempt 2: tab
    try:
        it = pd.read_csv(path, sep="\t", chunksize=CHUNK_ROWS)
        return it, "tab"
    except Exception:
        pass
    # Attempt 3: python engine auto-sep
    it = pd.read_csv(path, sep=None, engine="python", chunksize=CHUNK_ROWS)
    return it, "auto"


# =========================
# main fit
# =========================
def fit_azure_profile():
    azure_file = _auto_find_azure_file()
    chunk_iter, dialect = _iter_chunks(azure_file)

    print(f"[AzureFit] Using input: {azure_file} (dialect={dialect}, chunk_rows={CHUNK_ROWS})")

    # We aggregate arrays in lists to avoid huge memory spikes per pair
    gaps_list_by_pair: Dict[str, List[np.ndarray]] = {}
    durs_list_by_pair: Dict[str, List[np.ndarray]] = {}
    pair_counts: Dict[str, int] = {}

    ts_scale: Optional[float] = None
    dur_scale: Optional[float] = None

    for chunk in chunk_iter:
        # normalize column names (strip)
        chunk.columns = [str(c).strip() for c in chunk.columns]

        missing = [c for c in REQ_COLS if c not in chunk.columns]
        if missing:
            raise RuntimeError(f"[AzureFit] Missing columns {missing}. Expected {REQ_COLS}")

        df = chunk[REQ_COLS].dropna(subset=REQ_COLS).copy()

        df["app"] = df["app"].astype(str)
        df["func"] = df["func"].astype(str)
        df["end_timestamp"] = pd.to_numeric(df["end_timestamp"], errors="coerce")
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df = df.dropna(subset=["end_timestamp", "duration"])

        if df.empty:
            continue

        # infer scales once on the first non-empty chunk
        if ts_scale is None or dur_scale is None:
            ts_vals = df["end_timestamp"].to_numpy(dtype=np.float64)
            dur_vals = df["duration"].to_numpy(dtype=np.float64)
            ts_scale = _infer_ts_scale(ts_vals, TIME_UNIT)
            dur_scale = _infer_dur_scale(dur_vals, DURATION_UNIT)

        assert ts_scale is not None and dur_scale is not None

        df["end_s"] = df["end_timestamp"].astype(np.float64) * ts_scale
        df["dur_s"] = df["duration"].astype(np.float64) * dur_scale

        # realism fixes
        if DROP_NONPOSITIVE_DURATION:
            df = df[df["dur_s"] > 0].copy()
        if CLAMP_MIN_DURATION_S and CLAMP_MIN_DURATION_S > 0:
            df["dur_s"] = np.maximum(df["dur_s"].to_numpy(dtype=np.float64), float(CLAMP_MIN_DURATION_S))

        if df.empty:
            continue

        df["pair"] = df["app"] + "|" + df["func"]

        # counts
        vc = df["pair"].value_counts()
        for k, v in vc.items():
            pair_counts[k] = pair_counts.get(k, 0) + int(v)

        # per pair arrays
        for pair, g in df.groupby("pair"):
            # inter-arrival gaps based on end timestamps
            t = np.sort(g["end_s"].to_numpy(dtype=np.float64))
            gaps = np.diff(t) if t.size >= 2 else np.array([], dtype=np.float64)

            durs = g["dur_s"].to_numpy(dtype=np.float64)
            durs = durs[np.isfinite(durs)]

            if pair not in gaps_list_by_pair:
                gaps_list_by_pair[pair] = []
                durs_list_by_pair[pair] = []

            if gaps.size:
                gaps_list_by_pair[pair].append(gaps)
            if durs.size:
                durs_list_by_pair[pair].append(durs)

    if not pair_counts:
        raise RuntimeError("[AzureFit] No valid rows found after parsing/cleaning.")

    # finalize scales if file was empty early
    if ts_scale is None:
        ts_scale = 1.0
    if dur_scale is None:
        dur_scale = 1.0

    # sort pairs by count
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
    top_pairs = set([k for k, _ in sorted_pairs[:max(1, TOPK_PAIRS)]])

    profile: Dict[str, Any] = {
        "source": "Azure2021_simplified_app_func_end_duration",
        "input_file": str(azure_file),
        "schema": REQ_COLS,
        "time_scale_to_seconds": float(ts_scale),
        "duration_scale_to_seconds": float(dur_scale),
        "idle_threshold_s": float(IDLE_THRESHOLD_S),
        "topk_pairs": int(TOPK_PAIRS),
        "pairs": {},
        "other_aggregate": {},
        "global": {},
        "notes": {
            "dialect": dialect,
            "chunk_rows": CHUNK_ROWS,
            "drop_nonpositive_duration": bool(DROP_NONPOSITIVE_DURATION),
            "clamp_min_duration_s": float(CLAMP_MIN_DURATION_S) if CLAMP_MIN_DURATION_S else 0.0,
            "cold_prob_definition": f"P(inter_arrival_gap_s > {IDLE_THRESHOLD_S})",
        }
    }

    global_gaps = []
    global_durs = []
    other_gaps = []
    other_durs = []

    def _concat(lst: List[np.ndarray]) -> np.ndarray:
        if not lst:
            return np.array([], dtype=np.float64)
        return np.concatenate(lst).astype(np.float64)

    for pair, cnt in sorted_pairs:
        gaps = _concat(gaps_list_by_pair.get(pair, []))
        durs = _concat(durs_list_by_pair.get(pair, []))

        cold_prob = float(np.mean(gaps > IDLE_THRESHOLD_S)) if gaps.size else None

        if pair in top_pairs:
            profile["pairs"][pair] = {
                "count": int(cnt),
                "inter_arrival_gap_s": _stats(gaps),
                "duration_s": _stats(durs),
                "inferred_cold_prob": cold_prob,
            }
        else:
            if gaps.size:
                other_gaps.append(gaps)
            if durs.size:
                other_durs.append(durs)

        if gaps.size:
            global_gaps.append(gaps)
        if durs.size:
            global_durs.append(durs)

    # global stats
    if global_gaps:
        gg = np.concatenate(global_gaps)
        profile["global"]["inter_arrival_gap_s"] = _stats(gg)
        profile["global"]["inferred_cold_prob"] = float(np.mean(gg > IDLE_THRESHOLD_S))
    else:
        profile["global"]["inter_arrival_gap_s"] = _stats(np.array([], dtype=np.float64))
        profile["global"]["inferred_cold_prob"] = None

    if global_durs:
        gd = np.concatenate(global_durs)
        profile["global"]["duration_s"] = _stats(gd)
    else:
        profile["global"]["duration_s"] = _stats(np.array([], dtype=np.float64))

    # other aggregate
    if other_gaps:
        og = np.concatenate(other_gaps)
        profile["other_aggregate"]["inter_arrival_gap_s"] = _stats(og)
        profile["other_aggregate"]["inferred_cold_prob"] = float(np.mean(og > IDLE_THRESHOLD_S))
    if other_durs:
        od = np.concatenate(other_durs)
        profile["other_aggregate"]["duration_s"] = _stats(od)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"[AzureFit][OK] Wrote: {OUT_JSON}")
    print(f"[AzureFit] total_pairs={len(sorted_pairs)}, top_pairs_saved={min(TOPK_PAIRS, len(sorted_pairs))}")
    print(f"[AzureFit] global inferred cold prob={profile['global'].get('inferred_cold_prob', None)}")


if __name__ == "__main__":
    fit_azure_profile()
