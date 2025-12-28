import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ============================================================
# 配置：输出文件夹
# ============================================================
OUTPUT_DIR = "figures_effectiveness"
TARGET_FILE = "metrics_full.csv"  # 默认读取完整方法的日志


# ============================================================
# 0. 顶会图表风格 (Science/IEEE)
# ============================================================
def set_style():
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'legend.frameon': False,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'savefig.bbox': 'tight',
    })


# ============================================================
# 1. 绘图逻辑
# ============================================================
def save_plot(name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    print(f"Saved {name}")


def load_data():
    if not os.path.exists(TARGET_FILE):
        print(f"Error: {TARGET_FILE} not found. Please run the 'full' experiment first.")
        return None

    df = pd.read_csv(TARGET_FILE)
    # 平滑处理，让曲线更美观 (Window=20)
    smooth_cols = ['loss', 'acc_top1', 'tokens_per_s', 'step_time_ms',
                   'grad_bytes_hot', 'grad_bytes_cold', 'hot_ratio', 'grad_mode_hot_frac']
    for c in smooth_cols:
        if c in df.columns:
            df[f'{c}_smooth'] = df[c].rolling(20, min_periods=1).mean()
    return df


# --- Fig 1: Validity (Loss & Acc) ---
def plot_fig1_validity(df):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    color = '#D62728'  # Red for Loss
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(df['step'], df['loss_smooth'], color=color, linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()  # 双坐标轴
    color = '#1F77B4'  # Blue for Acc
    ax2.set_ylabel('Top-1 Accuracy', color=color, fontweight='bold')
    ax2.plot(df['step'], df['acc_top1_smooth'], color=color, linewidth=2, linestyle='--', label='Top-1 Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    plt.title("Fig 1: Validity (Convergence)")

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')

    save_plot("Fig1_Validity_Loss_Acc")


# --- Fig 2: Performance (Throughput & Latency) ---
def plot_fig2_performance(df):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    color = '#2CA02C'  # Green for Throughput
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Throughput (tokens/s)', color=color, fontweight='bold')
    ax1.plot(df['step'], df['tokens_per_s_smooth'], color=color, linewidth=2, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#FF7F0E'  # Orange for Latency
    ax2.set_ylabel('Step Latency (ms)', color=color, fontweight='bold')
    ax2.plot(df['step'], df['step_time_ms_smooth'], color=color, linewidth=2, linestyle='--', label='Latency')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Fig 2: Performance (SLA Metrics)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')

    save_plot("Fig2_Performance_Throughput_Latency")


# --- Fig 3: Efficiency (Grad Bytes Hot vs Cold) ---
def plot_fig3_efficiency(df):
    plt.figure(figsize=(7, 5))

    # 堆叠图展示带宽组成
    plt.stackplot(df['step'],
                  df['grad_bytes_hot_smooth'] / 1024 ** 2,  # Convert to MB
                  df['grad_bytes_cold_smooth'] / 1024 ** 2,
                  labels=['Hot Gradients (Optimized)', 'Cold Gradients (Standard)'],
                  colors=['#9467BD', '#C5B0D5'], alpha=0.8)

    plt.xlabel('Training Step')
    plt.ylabel('Gradient Comm Volume (MB)')
    plt.title("Fig 3: Efficiency (Communication Breakdown)")
    plt.legend(loc='upper left')
    save_plot("Fig3_Efficiency_Comm_Bytes")


# --- Fig 4: Adaptivity (Hot Ratio vs Strategy) ---
def plot_fig4_adaptivity(df):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    color = '#8C564B'  # Brown for Environment (Hot Ratio)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Active Hot Expert Ratio', color=color, fontweight='bold')
    # 原始数据可能波动大，用填充图表示趋势
    ax1.fill_between(df['step'], 0, df['hot_ratio_smooth'], color=color, alpha=0.2)
    ax1.plot(df['step'], df['hot_ratio_smooth'], color=color, linewidth=1.5, label='Env: Hot Ratio')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    color = '#E377C2'  # Pink for Strategy (Hot Mode Frac)
    ax2.set_ylabel('Hot-Path Dispatch Fraction', color=color, fontweight='bold')
    ax2.plot(df['step'], df['grad_mode_hot_frac_smooth'], color=color, linewidth=2.5, label='Policy: Hot Path')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0)

    plt.title("Fig 4: Adaptivity (Scheduler Response)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right')

    save_plot("Fig4_Adaptivity_Scheduler")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    set_style()
    print(">>> Loading Data for Effectiveness Analysis...")
    df = load_data()

    if df is not None:
        plot_fig1_validity(df)
        plot_fig2_performance(df)
        plot_fig3_efficiency(df)
        plot_fig4_adaptivity(df)
        print(f"\n>>> All effectiveness plots saved to ./{OUTPUT_DIR}")