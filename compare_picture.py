# -*- coding: utf-8 -*-
"""
ICWS-style ablation comparison figures.

What this script does (paper-ready + ablation-friendly):
- If 'ablation_mode' exists in metrics.csv:
    (A) Acc@5 vs Step (train curve + val scatter) grouped by ablation_mode
    (B) End-to-End Step Time vs Step grouped by ablation_mode
    (C) Total Cost (USD/step) vs Step grouped by ablation_mode
    (D) Deadline Miss Rate vs Step grouped by ablation_mode
    (E) Cost Breakdown (stacked bar) averaged over last K train steps per mode
    (F) Grad selector / mode fractions (optional bar) averaged over last K train steps per mode
- Otherwise: falls back to your previous "single run" plots (minimal set).

Style goals (typical ICWS / IEEE figures):
- clean serif font, compact sizes, thin axes, readable legends
- colorblind-friendly palette (Okabe-Ito) + line styles for grayscale printing

Run (PowerShell):
  python .\picture_ablation.py --csv .\metrics.csv --out_dir .\figures_icws --col 1 --smooth 10 --tail_k 200 --fmt pdf,png
"""

import os
import argparse
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Paper style (ICWS/IEEE-like)
# -------------------------
def set_paper_style(font="Times New Roman", base_fontsize=8):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [font, "Times New Roman", "Times", "DejaVu Serif"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "figure.titlesize": base_fontsize,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,
    })


def fig_size(col=1, aspect=0.72):
    # IEEE two-column typical widths: single ~3.5", double ~7.16"
    w = 3.5 if col == 1 else 7.16
    h = w * aspect
    return (w, h)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def rolling_mean(s: pd.Series, window: int):
    if window is None or window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def save_fig(fig, out_dir: str, name: str, fmt=("pdf", "png"), dpi=600):
    fig.tight_layout()
    for f in fmt:
        f = f.lower().strip()
        if not f:
            continue
        path = os.path.join(out_dir, f"{name}.{f}")
        if f == "png":
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.close(fig)


def get_phase(df: pd.DataFrame, phase: str) -> Optional[pd.DataFrame]:
    if "phase" not in df.columns:
        return None
    sub = df[df["phase"].astype(str).str.lower() == phase.lower()].copy()
    return sub if len(sub) > 0 else None


def fallback_train(df: pd.DataFrame) -> pd.DataFrame:
    train = get_phase(df, "train")
    if train is None:
        train = df.copy()
    return train


def to_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# -------------------------
# ICWS-friendly palette (Okabe-Ito) + robust line styles
# -------------------------
# Source reference: Okabe-Ito palette is widely used for colorblind-friendly scientific figures.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#000000",  # black
]

MODE_LABEL = {
    "full": "Full",
    "no_hotcold": "w/o HotCold",
    "sync_update": "SyncUpdate",
    "heuristic_only": "HeuristicOnly",
    "predictor_only": "PredictorOnly",
    "no_nsga2": "w/o NSGA-II",
}

MODE_ORDER = ["full", "no_hotcold", "sync_update", "heuristic_only", "predictor_only", "no_nsga2"]

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]  # diverse styles
MARKERS = ["o", "s", "^", "D", "v", "x"]


def mode_style(i: int) -> Dict[str, object]:
    color = OKABE_ITO[i % len(OKABE_ITO)]
    ls = LINESTYLES[i % len(LINESTYLES)]
    mk = MARKERS[i % len(MARKERS)]
    return {"color": color, "linestyle": ls, "marker": None, "markersize": 3, "markevery": 30}


def stable_sort(df: pd.DataFrame, x: str) -> pd.DataFrame:
    if x in df.columns:
        return df.sort_values([x]).reset_index(drop=True)
    return df


def split_by_mode(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    if "ablation_mode" not in df.columns:
        return [("full", df)]
    modes = [m for m in MODE_ORDER if m in set(df["ablation_mode"].astype(str).str.lower())]
    if not modes:
        modes = sorted(df["ablation_mode"].astype(str).str.lower().unique().tolist())
    out = []
    for m in modes:
        sub = df[df["ablation_mode"].astype(str).str.lower() == m].copy()
        out.append((m, sub))
    return out


def tail_mean(train: pd.DataFrame, col: str, tail_k: int) -> float:
    if col not in train.columns:
        return float("nan")
    s = pd.to_numeric(train[col], errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    if tail_k is not None and tail_k > 0:
        s = s.iloc[-tail_k:]
    return float(s.mean())


# -------------------------
# Ablation comparison plots
# -------------------------
def plot_acc5_ablation(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    if "acc_top5" not in df.columns:
        print("[skip] acc@5: acc_top5 not found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=0.72))
    ax = fig.add_subplot(111)

    for i, (mode, sub) in enumerate(split_by_mode(df)):
        sub = stable_sort(sub, x)
        train = get_phase(sub, "train")
        val = get_phase(sub, "val")
        st = mode_style(i)
        label = MODE_LABEL.get(mode, mode)

        if train is not None and len(train) > 0:
            ax.plot(train[x], rolling_mean(train["acc_top5"], smooth), label=label, **st)

        # val points (optional)
        if val is not None and len(val) > 0:
            ax.scatter(val[x], val["acc_top5"], s=10, marker="o", color=st["color"], alpha=0.9)

    ax.set_xlabel(x)
    ax.set_ylabel("Accuracy@5")
    ax.set_title("Acc@5 vs Step (Ablation)")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_dir, "ablation_acc5_vs_step", fmt=fmt, dpi=dpi)


def plot_step_time_ablation(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    if "step_time_ms" not in df.columns:
        print("[skip] step time: step_time_ms not found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=0.72))
    ax = fig.add_subplot(111)

    for i, (mode, sub) in enumerate(split_by_mode(df)):
        sub = stable_sort(sub, x)
        train = get_phase(sub, "train") or sub
        st = mode_style(i)
        label = MODE_LABEL.get(mode, mode)

        ax.plot(train[x], rolling_mean(train["step_time_ms"], smooth), label=label, **st)

    ax.set_xlabel(x)
    ax.set_ylabel("Step Time (ms)")
    ax.set_title("End-to-End Step Time vs Step (Ablation)")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_dir, "ablation_step_time_vs_step", fmt=fmt, dpi=dpi)


def plot_total_cost_ablation(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    if "cost_usd_step" not in df.columns:
        print("[skip] total cost: cost_usd_step not found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=0.72))
    ax = fig.add_subplot(111)

    for i, (mode, sub) in enumerate(split_by_mode(df)):
        sub = stable_sort(sub, x)
        train = get_phase(sub, "train") or sub
        st = mode_style(i)
        label = MODE_LABEL.get(mode, mode)

        ax.plot(train[x], rolling_mean(train["cost_usd_step"], smooth), label=label, **st)

    ax.set_xlabel(x)
    ax.set_ylabel("Cost (USD / step)")
    ax.set_title("Total Cost vs Step (Ablation)")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_dir, "ablation_total_cost_vs_step", fmt=fmt, dpi=dpi)


def plot_deadline_miss_ablation(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    if "deadline_miss" not in df.columns:
        print("[skip] deadline miss: deadline_miss not found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=0.72))
    ax = fig.add_subplot(111)

    for i, (mode, sub) in enumerate(split_by_mode(df)):
        sub = stable_sort(sub, x)
        train = get_phase(sub, "train") or sub
        st = mode_style(i)
        label = MODE_LABEL.get(mode, mode)

        ax.plot(train[x], rolling_mean(train["deadline_miss"].fillna(0.0), smooth), label=label, **st)

    ax.set_xlabel(x)
    ax.set_ylabel("Miss Rate (rolling mean)")
    ax.set_title("Deadline Miss Rate vs Step (Ablation)")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_dir, "ablation_deadline_miss_vs_step", fmt=fmt, dpi=dpi)


def plot_cost_breakdown_bar(df, out_dir, tail_k=200, col=1, dpi=600, fmt=("pdf", "png")):
    # stacked bar of average cost components over last K train steps per mode
    comps = [
        "cost_usd_pre_fwd",
        "cost_usd_post_fwd",
        "cost_usd_expert_fwd",
        "cost_usd_pre_bwd",
        "cost_usd_post_bwd",
        "cost_usd_grad_apply",
    ]
    if not any(c in df.columns for c in comps):
        print("[skip] cost breakdown bar: cost components not found.")
        return

    label_map = {
        "cost_usd_pre_fwd": "pre_fwd",
        "cost_usd_post_fwd": "post_fwd",
        "cost_usd_expert_fwd": "expert_fwd",
        "cost_usd_pre_bwd": "pre_bwd",
        "cost_usd_post_bwd": "post_bwd",
        "cost_usd_grad_apply": "grad_apply",
    }

    modes = split_by_mode(df)
    names = [MODE_LABEL.get(m, m) for m, _ in modes]
    vals = {c: [] for c in comps if c in df.columns}

    for mode, sub in modes:
        train = get_phase(sub, "train") or sub
        train = train.copy()
        for c in vals.keys():
            vals[c].append(tail_mean(train, c, tail_k))

    x = np.arange(len(names))
    fig = plt.figure(figsize=fig_size(col, aspect=0.62))
    ax = fig.add_subplot(111)

    bottom = np.zeros(len(names), dtype=float)
    # component colors: use muted subset of Okabe-Ito for stack
    comp_colors = {
        "cost_usd_pre_fwd": "#56B4E9",
        "cost_usd_post_fwd": "#0072B2",
        "cost_usd_expert_fwd": "#009E73",
        "cost_usd_pre_bwd": "#E69F00",
        "cost_usd_post_bwd": "#D55E00",
        "cost_usd_grad_apply": "#CC79A7",
    }

    for c in vals.keys():
        y = np.array(vals[c], dtype=float)
        ax.bar(x, y, bottom=bottom, label=label_map.get(c, c), color=comp_colors.get(c, None))
        bottom += np.nan_to_num(y, nan=0.0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(f"Avg Cost (USD/step, last {tail_k})")
    ax.set_title("Cost Breakdown (Ablation, Tail Average)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=3)
    save_fig(fig, out_dir, "ablation_cost_breakdown_bar", fmt=fmt, dpi=dpi)


def plot_grad_mode_bar(df, out_dir, tail_k=200, col=1, dpi=600, fmt=("pdf", "png")):
    cols = ["grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac"]
    if not any(c in df.columns for c in cols):
        print("[skip] grad mode bar: grad_mode_* not found.")
        return

    modes = split_by_mode(df)
    names = [MODE_LABEL.get(m, m) for m, _ in modes]

    # averages per mode
    data = []
    for mode, sub in modes:
        train = get_phase(sub, "train") or sub
        row = []
        for c in cols:
            row.append(tail_mean(train, c, tail_k))
        data.append(row)

    data = np.array(data, dtype=float)  # [M, 3]
    x = np.arange(len(names))
    width = 0.24

    fig = plt.figure(figsize=fig_size(col, aspect=0.62))
    ax = fig.add_subplot(111)

    labels = ["hot", "cold", "http"]
    colors = ["#009E73", "#E69F00", "#0072B2"]  # green/orange/blue
    for j in range(3):
        ax.bar(x + (j - 1) * width, data[:, j], width=width, label=labels[j], color=colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(f"Avg Fraction (last {tail_k})")
    ax.set_ylim(0, 1.0)
    ax.set_title("Grad Apply Mode Fractions (Ablation, Tail Average)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=3)
    save_fig(fig, out_dir, "ablation_grad_mode_bar", fmt=fmt, dpi=dpi)


# -------------------------
# Fallback: single-run plots (kept minimal)
# -------------------------
def plot_acc5_single(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    if "acc_top5" not in df.columns:
        print("[skip] acc@5: acc_top5 not found.")
        return
    train = get_phase(df, "train")
    val = get_phase(df, "val")

    fig = plt.figure(figsize=fig_size(col, aspect=0.72))
    ax = fig.add_subplot(111)

    if train is not None:
        ax.plot(train[x], rolling_mean(train["acc_top5"], smooth), label="train", color=OKABE_ITO[0], linestyle="-")
    if val is not None:
        ax.scatter(val[x], val["acc_top5"], label="val", s=10, marker="o", color=OKABE_ITO[1])

    ax.set_xlabel(x)
    ax.set_ylabel("Accuracy@5")
    ax.set_title("Accuracy@5 vs Step")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, out_dir, "acc5_vs_step", fmt=fmt, dpi=dpi)


def plot_step_time_single(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    if "step_time_ms" not in df.columns:
        print("[skip] step time: step_time_ms not found.")
        return
    train = fallback_train(df)
    fig = plt.figure(figsize=fig_size(col, aspect=0.72))
    ax = fig.add_subplot(111)
    ax.plot(train[x], rolling_mean(train["step_time_ms"], smooth), label="step_time", color=OKABE_ITO[0])
    ax.set_xlabel(x)
    ax.set_ylabel("Step Time (ms)")
    ax.set_title("End-to-End Step Time vs Step")
    ax.grid(True)
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "step_time_vs_step", fmt=fmt, dpi=dpi)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="metrics.csv")
    ap.add_argument("--out_dir", type=str, default="figures_icws")
    ap.add_argument("--col", type=int, default=1, choices=[1, 2])
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--tail_k", type=int, default=200, help="tail window size for bar summaries")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--fmt", type=str, default="pdf,png")
    ap.add_argument("--x", type=str, default="step", choices=["step", "epoch", "step_in_epoch"])
    ap.add_argument("--font", type=str, default="Times New Roman")
    ap.add_argument("--fontsize", type=int, default=8)
    args = ap.parse_args()

    set_paper_style(font=args.font, base_fontsize=args.fontsize)
    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.csv)

    # numeric coercion for required cols
    needed = [
        args.x, "loss", "acc_top5", "step_time_ms",
        "deadline_miss",
        "cost_usd_step",
        "cost_usd_pre_fwd", "cost_usd_post_fwd", "cost_usd_expert_fwd",
        "cost_usd_pre_bwd", "cost_usd_post_bwd", "cost_usd_grad_apply",
        "grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac",
    ]
    to_numeric(df, needed)

    # normalize mode strings
    if "ablation_mode" in df.columns:
        df["ablation_mode"] = df["ablation_mode"].astype(str).str.lower()

    df = stable_sort(df, args.x)
    fmt = tuple([s.strip().lower() for s in args.fmt.split(",") if s.strip()])

    has_ablation = "ablation_mode" in df.columns and df["ablation_mode"].nunique() > 1

    if has_ablation:
        plot_acc5_ablation(df, args.out_dir, x=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
        plot_step_time_ablation(df, args.out_dir, x=args.x, col=args.col, smooth=max(args.smooth, 10), dpi=args.dpi, fmt=fmt)
        plot_total_cost_ablation(df, args.out_dir, x=args.x, col=args.col, smooth=max(args.smooth, 10), dpi=args.dpi, fmt=fmt)
        plot_deadline_miss_ablation(df, args.out_dir, x=args.x, col=args.col, smooth=max(args.smooth, 10), dpi=args.dpi, fmt=fmt)
        plot_cost_breakdown_bar(df, args.out_dir, tail_k=max(args.tail_k, 50), col=args.col, dpi=args.dpi, fmt=fmt)
        plot_grad_mode_bar(df, args.out_dir, tail_k=max(args.tail_k, 50), col=args.col, dpi=args.dpi, fmt=fmt)
    else:
        # fallback single-run minimal
        plot_acc5_single(df, args.out_dir, x=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
        plot_step_time_single(df, args.out_dir, x=args.x, col=args.col, smooth=max(args.smooth, 10), dpi=args.dpi, fmt=fmt)


if __name__ == "__main__":
    main()
