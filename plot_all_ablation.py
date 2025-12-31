import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
from matplotlib.ticker import PercentFormatter

OUTPUT_DIR = "figures_ablation"


# ============================================================
# 0. 顶会风格配置 (IEEE/ACM Transaction Style)
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
        'legend.frameon': False,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })


# 颜色策略：Proposed 用最醒目的红色，Baseline 用冷色调
COLOR_MAP = {
    "Proposed (Full)": "#D62728",  # Bold Red
    "w/o Hot/Cold": "#1F77B4",  # Blue
    "Heuristic Only": "#FF7F0E",  # Orange
    "Predictor Only": "#2CA02C",  # Green
    "w/o NSGA-II": "#9467BD",  # Purple
    "Sync Update": "#7F7F7F"  # Grey
}

# 线型策略
STYLE_MAP = {
    "Proposed (Full)": {"ls": "-", "lw": 3, "zorder": 10},
    "w/o Hot/Cold": {"ls": "--", "lw": 2, "zorder": 5},
    "Heuristic Only": {"ls": "-.", "lw": 2, "zorder": 4},
    "Predictor Only": {"ls": ":", "lw": 2, "zorder": 3},
    "w/o NSGA-II": {"ls": "--", "lw": 2, "zorder": 2},
    "Sync Update": {"ls": "-.", "lw": 2, "zorder": 1}
}


def save_plot(name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    print(f"Saved {name}")


# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_ablation_data():
    name_map = {
        "metrics_full.csv": "Proposed (Full)",
        "metrics_no_hotcold.csv": "w/o Hot/Cold",
        "metrics_heuristic_only.csv": "Heuristic Only",
        "metrics_predictor_only.csv": "Predictor Only",
        "metrics_no_nsga2.csv": "w/o NSGA-II",
        "metrics_sync_update.csv": "Sync Update"
    }
    dfs = []
    for f in glob.glob("metrics_*.csv"):
        try:
            df = pd.read_csv(f)
            df['Method'] = name_map.get(f, f.replace("metrics_", "").replace(".csv", ""))

            # 预计算 P99 (Rolling window for stability curve)
            df['p99_latency'] = df['step_time_ms'].rolling(50).quantile(0.99)

            # 平滑均值
            for c in ['step_time_ms', 'loss', 'inv_cold_ms', 'cost_usd_step', 'tokens_per_s']:
                if c in df.columns:
                    df[f'{c}_smooth'] = df[c].rolling(20, min_periods=1).mean()
            dfs.append(df)
        except:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================
# 2. 强力证明图表绘制
# ============================================================

def plot_cdf_latency(df):
    """ 图1: 延迟 CDF (P99 优势的最强证明) """
    plt.figure(figsize=(7, 5))

    methods = sorted(df['Method'].unique())
    # Proposed 最后画
    if "Proposed (Full)" in methods:
        methods.remove("Proposed (Full)")
        methods.append("Proposed (Full)")

    for m in methods:
        # 取后半段稳定数据绘制 CDF，避免初期冷启动干扰
        subset = df[df['Method'] == m]
        stable_subset = subset[subset['step'] > subset['step'].max() * 0.2]
        data = stable_subset['step_time_ms'].sort_values()
        y = np.arange(1, len(data) + 1) / len(data)

        plt.plot(data, y,
                 label=m,
                 color=COLOR_MAP.get(m, 'k'),
                 linestyle=STYLE_MAP.get(m, {}).get('ls', '-'),
                 linewidth=STYLE_MAP.get(m, {}).get('lw', 2))

    plt.xlabel("End-to-End Latency (ms)")
    plt.ylabel("Cumulative Probability (CDF)")
    plt.title("Latency Distribution (Tail Latency)")
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    # 聚焦于 P50-P100 区间，展示尾部差异
    plt.ylim(0, 1.02)
    # x轴截断到 99 分位数的 1.2 倍，去掉极端异常值让图更紧凑
    limit = df['step_time_ms'].quantile(0.98)
    plt.xlim(0, limit)

    plt.legend(loc='lower right')
    save_plot("Ablation_CDF_Latency")


def plot_normalized_bar(df):
    """ 图2: 归一化性能提升 (Bar Chart) - 论文最喜欢的 Summary 图 """
    plt.figure(figsize=(8, 5))

    # 选取后 50% 的步骤计算稳定均值
    stable_df = df[df['step'] > df['step'].max() * 0.5]

    metrics = {
        'step_time_ms': 'Avg Latency',
        'cost_usd_step': 'Training Cost',
        'inv_cold_ms': 'Cold Overhead'
    }

    summary = stable_df.groupby('Method')[list(metrics.keys())].mean().reset_index()

    # 以 Heuristic Only (或 w/o Hot/Cold) 为基准 (Baseline = 1.0)
    baseline_method = "Heuristic Only"
    if baseline_method not in summary['Method'].values:
        baseline_method = "w/o Hot/Cold"  # Fallback

    baseline_row = summary[summary['Method'] == baseline_method].iloc[0]

    plot_data = []
    methods_to_plot = [m for m in summary['Method'].unique()]

    # 重新排序：Baseline 第一，Proposed 最后
    if baseline_method in methods_to_plot:
        methods_to_plot.remove(baseline_method)
    if "Proposed (Full)" in methods_to_plot:
        methods_to_plot.remove("Proposed (Full)")
    methods_to_plot = [baseline_method] + sorted(methods_to_plot) + ["Proposed (Full)"]

    # 准备绘图数据 (Melting)
    bar_width = 0.2
    x = np.arange(len(metrics))

    ax = plt.gca()

    for i, method in enumerate(methods_to_plot):
        row = summary[summary['Method'] == method].iloc[0]
        # 计算归一化值 (Baseline = 1.0)
        norm_values = [row[m] / baseline_row[m] for m in metrics.keys()]

        offset = (i - len(methods_to_plot) / 2) * bar_width + bar_width / 2

        bars = ax.bar(x + offset, norm_values, width=bar_width,
                      label=method, color=COLOR_MAP.get(method, 'gray'), edgecolor='white')

        # 在 Proposed 的柱子上标数值
        if method == "Proposed (Full)":
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics.values())
    ax.set_ylabel(f"Normalized Metric (Lower is Better)\nBaseline: {baseline_method}")
    ax.set_title("Performance Improvement vs. Baseline")

    # 画一条 y=1.0 的基准线
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    save_plot("Ablation_Normalized_Bar")


def plot_cost_perf_scatter(df):
    """ 图3: 增强版 Trade-off 散点图 """
    plt.figure(figsize=(7, 5))
    summary = df.groupby('Method').agg({'step_time_ms': 'mean', 'cost_usd_step': 'mean'}).reset_index()

    for _, row in summary.iterrows():
        m = row['Method']
        c = COLOR_MAP.get(m, 'gray')

        # Proposed 也是大星星
        if m == "Proposed (Full)":
            plt.scatter(row['step_time_ms'], row['cost_usd_step'],
                        c=c, marker='*', s=350, edgecolors='k', zorder=20, label=m)
        else:
            plt.scatter(row['step_time_ms'], row['cost_usd_step'],
                        c=c, marker='o', s=120, edgecolors='white', zorder=10, label=m, alpha=0.8)

        # 文字标注
        offset_y = row['cost_usd_step'] * 0.02
        fw = 'bold' if m == "Proposed (Full)" else 'normal'
        plt.text(row['step_time_ms'], row['cost_usd_step'] + offset_y, m,
                 ha='center', va='bottom', fontsize=10, fontweight=fw, color=c)

    plt.xlabel("Avg Latency (ms) [Lower is Better]")
    plt.ylabel("Avg Cost ($) [Lower is Better]")
    plt.title("Cost-Performance Pareto Frontier")
    plt.grid(True, linestyle='--')

    # 绘制 Pareto 区域背景 (可选)
    xlim, ylim = plt.xlim(), plt.ylim()
    # 假设左下角是最优区

    save_plot("Ablation_Tradeoff_Pareto")


def plot_throughput_comparison(df):
    """ 图4: 吞吐量对比曲线 (Throughput) """
    plt.figure(figsize=(7, 5))

    methods = sorted(df['Method'].unique())
    if "Proposed (Full)" in methods:
        methods.remove("Proposed (Full)")
        methods.append("Proposed (Full)")

    for m in methods:
        sub = df[df['Method'] == m]
        plt.plot(sub['step'], sub['tokens_per_s_smooth'],
                 label=m, color=COLOR_MAP.get(m),
                 ls=STYLE_MAP.get(m, {}).get('ls'), lw=STYLE_MAP.get(m, {}).get('lw'))

    plt.xlabel("Training Step")
    plt.ylabel("Throughput (Tokens/sec)")
    plt.title("System Throughput Stability")
    plt.legend(loc='lower right')  # 吞吐量通常随时间上升，图例放下面
    save_plot("Ablation_Throughput")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    set_style()
    print(">>> Loading Data for Stronger Ablation Proof...")
    df = load_ablation_data()

    if df is not None:
        # 1. CDF: 证明尾部延迟优势 (稳定性)
        plot_cdf_latency(df)

        # 2. Normalized Bar: 证明综合指标提升幅度 (x% better)
        plot_normalized_bar(df)

        # 3. Pareto Scatter: 证明性价比优势 (Dominance)
        plot_cost_perf_scatter(df)

        # 4. Throughput Curve: 证明持续吞吐能力
        plot_throughput_comparison(df)

        print(f"\n✅ Enhanced Ablation plots saved to ./{OUTPUT_DIR}")
        print("   - CDF Latency: Prove stability (short tail)")
        print("   - Normalized Bar: Quantify improvement (e.g., 0.6x cost)")
        print("   - Pareto Scatter: Prove dominance strategy")