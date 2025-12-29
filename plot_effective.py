import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = "figures_effectiveness"
TARGET_FILE = "metrics_full.csv"


# ============================================================
# 0. 顶会图表风格 (ICWS/IEEE Style)
# ============================================================
def set_style():
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'legend.frameon': False,  # 无边框图例
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,  # 网格更淡
        'savefig.bbox': 'tight',
        'axes.spines.top': False,  # 去掉顶部框线
        'axes.spines.right': False  # 去掉右侧框线
    })


def save_plot(name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    print(f"Saved {name}")


def load_data():
    if not os.path.exists(TARGET_FILE):
        print(f"[Error] {TARGET_FILE} not found. Please run experiments first.")
        return None

    df = pd.read_csv(TARGET_FILE)
    # 关键指标平滑 (Window=20)
    cols = ['loss', 'acc_top1', 'acc_top5', 'tokens_per_s', 'step_time_ms',
            'grad_bytes_hot', 'grad_bytes_cold', 'hot_ratio', 'grad_mode_hot_frac',
            'inv_compute_ms', 'inv_net_ms', 'inv_queue_ms', 'inv_cold_ms']

    for c in cols:
        if c in df.columns:
            df[f'{c}_smooth'] = df[c].rolling(20, min_periods=1).mean()
    return df


# ============================================================
# Fig 1: Validity (Convergence)
# 调整：图例放顶部，颜色加深
# ============================================================
def plot_fig1_validity(df):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Loss (Left, Red)
    color_loss = '#C0392B'  # Deep Red
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss', color=color_loss, fontweight='bold')
    l1, = ax1.plot(df['step'], df['loss_smooth'], color=color_loss, lw=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_ylim(bottom=0)

    # Top-5 Acc (Right, Blue)
    ax2 = ax1.twinx()
    color_acc = '#2980B9'  # Strong Blue
    ax2.set_ylabel('Top-5 Accuracy', color=color_acc, fontweight='bold')
    l2, = ax2.plot(df['step'], df['acc_top5_smooth'], color=color_acc, lw=2, ls='--', label='Top-5 Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(0, 1.05)

    # 去掉右侧多余边框
    ax2.spines['top'].set_visible(False)

    # 统一图例位置：上方居中
    plt.legend(handles=[l1, l2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.title("Fig 1: Training Validity", y=1.15)  # 标题上移

    save_plot("Fig1_Validity")


# ============================================================
# Fig 2: Performance (SLA)
# 调整：图例放顶部，颜色区分度高
# ============================================================
def plot_fig2_performance(df):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Throughput (Left, Green)
    color_tps = '#27AE60'  # Forest Green
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Throughput (tokens/s)', color=color_tps, fontweight='bold')
    l1, = ax1.plot(df['step'], df['tokens_per_s_smooth'], color=color_tps, lw=2, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color_tps)
    ax1.set_ylim(bottom=0)

    # Latency (Right, Orange)
    ax2 = ax1.twinx()
    color_lat = '#D35400'  # Pumpkin Orange
    ax2.set_ylabel('Step Latency (ms)', color=color_lat, fontweight='bold')
    l2, = ax2.plot(df['step'], df['step_time_ms_smooth'], color=color_lat, lw=2, ls='--', label='Latency')
    ax2.tick_params(axis='y', labelcolor=color_lat)
    ax2.set_ylim(bottom=0)
    ax2.spines['top'].set_visible(False)

    # 统一图例
    plt.legend(handles=[l1, l2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.title("Fig 2: System Performance", y=1.15)

    save_plot("Fig2_Performance")


# ============================================================
# Fig 3: Efficiency (Comm)
# 调整：颜色柔和，图例不遮挡
# ============================================================
def plot_fig3_efficiency(df):
    plt.figure(figsize=(7, 5))

    hot_mb = df['grad_bytes_hot_smooth'] / (1024 ** 2)
    cold_mb = df['grad_bytes_cold_smooth'] / (1024 ** 2)

    # 颜色：Hot (Purple), Cold (Grey/Blue)
    pal = ['#8E44AD', '#BDC3C7']

    plt.stackplot(df['step'], hot_mb, cold_mb,
                  labels=['Hot Gradients (Optimized)', 'Cold Gradients (Standard)'],
                  colors=pal, alpha=0.9)

    plt.xlabel('Training Step')
    plt.ylabel('Comm. Volume (MB)')

    # 图例放图内左上角 (通常 stackplot 左侧较低)
    plt.legend(loc='upper left')
    plt.title("Fig 3: Communication Efficiency")

    save_plot("Fig3_Efficiency")


# ============================================================
# Fig 4: Adaptivity (Scheduling)
# 调整：修复变量名，清晰展示 Fill 和 Line
# ============================================================
def plot_fig4_adaptivity(df):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Env (Brown Area)
    color_env = '#795548'  # Brown
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Env: Hot Ratio', color=color_env, fontweight='bold')
    ax1.fill_between(df['step'], 0, df['hot_ratio_smooth'], color=color_env, alpha=0.2)
    l1, = ax1.plot(df['step'], df['hot_ratio_smooth'], color=color_env, lw=1, alpha=0.6, label='Env: Hot Ratio')
    ax1.tick_params(axis='y', labelcolor=color_env)
    ax1.set_ylim(0, 1.0)

    # Policy (Pink Line)
    ax2 = ax1.twinx()
    color_pol = '#E91E63'  # Pink
    ax2.set_ylabel('Policy: Hot Path %', color=color_pol, fontweight='bold')
    l2, = ax2.plot(df['step'], df['grad_mode_hot_frac_smooth'], color=color_pol, lw=2.5, label='Policy: Hot Path')
    ax2.tick_params(axis='y', labelcolor=color_pol)
    ax2.set_ylim(0, 1.0)
    ax2.spines['top'].set_visible(False)

    # 复合图例
    legend_elements = [
        Patch(facecolor=color_env, alpha=0.2, label='Env: Hot Expert Ratio'),
        Line2D([0], [0], color=color_pol, lw=2.5, label='Policy: Hot Path Dispatch'),
    ]
    plt.legend(handles=legend_elements, loc='lower right')  # 右下角通常为空
    plt.title("Fig 4: Scheduler Adaptivity")

    save_plot("Fig4_Adaptivity")


# ============================================================
# Fig 5: Breakdown
# 调整：颜色更专业，图例放外侧
# ============================================================
def plot_fig5_breakdown(df):
    plt.figure(figsize=(7, 5))

    compute = df['inv_compute_ms_smooth'].fillna(0)
    net = df['inv_net_ms_smooth'].fillna(0)
    queue = df['inv_queue_ms_smooth'].fillna(0)
    cold = df['inv_cold_ms_smooth'].fillna(0)

    # 颜色：Compute(Grey), Net(Blue), Queue(Orange), Cold(Red)
    pal = ["#7F8C8D", "#3498DB", "#F39C12", "#E74C3C"]
    labels = ["Compute", "Network", "Queueing", "Cold Start"]

    plt.stackplot(df['step'], compute, net, queue, cold, labels=labels, colors=pal, alpha=0.9)

    plt.xlabel("Training Step")
    plt.ylabel("Latency Breakdown (ms)")

    # 图例横向排布于上方
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.title("Fig 5: Latency Overhead Analysis", y=1.15)

    save_plot("Fig5_Breakdown")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    set_style()
    print(">>> Generating Effectiveness Plots...")
    df = load_data()
    if df is not None:
        plot_fig1_validity(df)
        plot_fig2_performance(df)
        plot_fig3_efficiency(df)
        plot_fig4_adaptivity(df)
        plot_fig5_breakdown(df)
        print("\n✅ Effectiveness plots saved.")