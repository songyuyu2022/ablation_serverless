import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

# ============================================================
# 配置
# ============================================================
ROOT_OUTPUT_DIR = "figures_ablation_pairwise"


# 顶会风格配置
def set_style():
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 15,
        'legend.frameon': False,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })


# 颜色：Proposed 始终为红色，Baseline 为灰色/蓝色
COLOR_FULL = "#D62728"  # Red
COLOR_BASE = "#1F77B4"  # Blue (or Grey "#7F7F7F")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_plot(folder, filename):
    ensure_dir(os.path.join(ROOT_OUTPUT_DIR, folder))
    path_pdf = os.path.join(ROOT_OUTPUT_DIR, folder, f"{filename}.pdf")
    path_png = os.path.join(ROOT_OUTPUT_DIR, folder, f"{filename}.png")
    plt.savefig(path_pdf)
    plt.savefig(path_png)
    print(f"Saved: {path_pdf}")
    plt.close()


# ============================================================
# 数据加载
# ============================================================
def load_data():
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

            # 兼容性处理：总通信量
            if 'grad_bytes' in df.columns:
                df['total_bytes'] = df['grad_bytes']
            else:
                df['total_bytes'] = df.get('grad_bytes_hot', 0) + df.get('grad_bytes_cold', 0)

            # 平滑处理
            for c in ['step_time_ms', 'cost_usd_step', 'inv_cold_ms', 'tokens_per_s', 'total_bytes']:
                if c in df.columns:
                    df[f'{c}_smooth'] = df[c].rolling(20, min_periods=1).mean()

            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================
# 绘图逻辑：一对一对比
# ============================================================

def plot_pair_line(df, method_base, metric_col, metric_name, folder, title):
    """ 通用双曲线对比图 """
    plt.figure(figsize=(8, 6))

    # 筛选数据
    df_full = df[df['Method'] == "Proposed (Full)"]
    df_base = df[df['Method'] == method_base]

    if df_base.empty:
        print(f"[Warn] Baseline '{method_base}' not found.")
        return

    # 绘制 Baseline (虚线，冷色)
    plt.plot(df_base['step'], df_base[f'{metric_col}_smooth'],
             label=method_base, color=COLOR_BASE, linestyle='--', linewidth=2.5, alpha=0.8)

    # 绘制 Full (实线，红色，置顶)
    plt.plot(df_full['step'], df_full[f'{metric_col}_smooth'],
             label="Proposed (Full)", color=COLOR_FULL, linestyle='-', linewidth=3, zorder=10)

    plt.xlabel("Training Step")
    plt.ylabel(metric_name)
    plt.title(title)

    # 标注提升幅度 (在中间位置)
    mid_step = int(df_full['step'].max() * 0.7)
    val_full = df_full[df_full['step'] > mid_step][f'{metric_col}_smooth'].mean()
    val_base = df_base[df_base['step'] > mid_step][f'{metric_col}_smooth'].mean()

    if val_base != 0:
        diff = (val_full - val_base) / val_base * 100
        # 判断是升高好还是降低好
        is_lower_better = "Throughput" not in metric_name

        if (is_lower_better and diff < 0) or (not is_lower_better and diff > 0):
            arrow_text = f" Improvement: {abs(diff):.1f}%"
            # 在图中间画一个箭头或文字
            plt.annotate(arrow_text,
                         xy=(mid_step, (val_full + val_base) / 2),
                         xytext=(mid_step, val_base * 1.2 if is_lower_better else val_base * 0.8),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         fontsize=12, fontweight='bold', ha='center')

    plt.legend(loc='best')
    save_plot(folder, f"{metric_col}_vs_step")


def plot_pair_scatter(df, method_base, folder):
    """ 成本-性能权衡散点图 (Full vs Baseline) """
    plt.figure(figsize=(7, 6))

    df_full = df[df['Method'] == "Proposed (Full)"]
    df_base = df[df['Method'] == method_base]

    if df_base.empty: return

    # 计算均值点
    mean_full_lat = df_full['step_time_ms'].mean()
    mean_full_cost = df_full['cost_usd_step'].mean()
    mean_base_lat = df_base['step_time_ms'].mean()
    mean_base_cost = df_base['cost_usd_step'].mean()

    # 绘制大点
    plt.scatter(mean_base_lat, mean_base_cost, s=300, color=COLOR_BASE, marker='o', label=method_base, edgecolors='k',
                alpha=0.8)
    plt.scatter(mean_full_lat, mean_full_cost, s=400, color=COLOR_FULL, marker='*', label="Proposed (Full)",
                edgecolors='k', zorder=10)

    # 绘制所有点背景 (可选，显示分布)
    # plt.scatter(df_base['step_time_ms'], df_base['cost_usd_step'], color=COLOR_BASE, alpha=0.05, s=10)
    # plt.scatter(df_full['step_time_ms'], df_full['cost_usd_step'], color=COLOR_FULL, alpha=0.05, s=10)

    plt.xlabel("Avg Latency (ms) [Lower is Better]")
    plt.ylabel("Avg Cost ($) [Lower is Better]")
    plt.title(f"Cost-Performance: Full vs {method_base}")
    plt.legend()
    plt.grid(True, linestyle='--')

    save_plot(folder, "Tradeoff_Scatter")


def plot_pair_cdf(df, method_base, folder):
    """ 延迟 CDF 对比 (稳定性) """
    plt.figure(figsize=(8, 6))

    df_full = df[df['Method'] == "Proposed (Full)"]
    df_base = df[df['Method'] == method_base]

    if df_base.empty: return

    for label, sub_df, color, ls in [
        (method_base, df_base, COLOR_BASE, '--'),
        ("Proposed (Full)", df_full, COLOR_FULL, '-')
    ]:
        data = sub_df['step_time_ms'].sort_values()
        y = np.arange(1, len(data) + 1) / len(data)
        plt.plot(data, y, label=label, color=color, linestyle=ls, linewidth=3)

    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title(f"Latency Stability: Full vs {method_base}")
    plt.ylim(0, 1.05)

    # 截断 X 轴以展示细节
    limit = df_full['step_time_ms'].quantile(0.99) * 1.5
    plt.xlim(0, limit)

    plt.legend(loc="lower right")
    save_plot(folder, "Latency_CDF")


# ============================================================
# Main: 定义对比组
# ============================================================
if __name__ == "__main__":
    set_style()
    print(">>> Loading Data...")
    df = load_data()

    if df is not None:
        # -------------------------------------------------
        # 1. 对比 NSGA-II (证明智能调度有效性)
        # -------------------------------------------------
        # 核心指标：Cost (调度器能省钱), Latency (调度器能加速)
        folder = "1_Effect_of_NSGA2"
        baseline = "w/o NSGA-II"
        print(f"Generating {folder}...")

        # 图1: Cost vs Latency Tradeoff (最强证明)
        plot_pair_scatter(df, baseline, folder)
        # 图2: 成本随时间变化
        plot_pair_line(df, baseline, 'cost_usd_step', "Training Cost ($)", folder, "Cost Optimization Analysis")

        # -------------------------------------------------
        # 2. 对比 Hot/Cold (证明冷热优化有效性)
        # -------------------------------------------------
        # 核心指标：Cold Start (冷启动), Total Bytes (通信)
        folder = "2_Effect_of_HotCold"
        baseline = "w/o Hot/Cold"
        print(f"Generating {folder}...")

        # 图1: 冷启动延迟曲线 (证明 Full 几乎无冷启动)
        plot_pair_line(df, baseline, 'inv_cold_ms', "Cold Start Latency (ms)", folder, "Cold Start Mitigation")
        # 图2: 通信量对比 (证明 Full 节省带宽)
        plot_pair_line(df, baseline, 'total_bytes', "Comm. Volume (Bytes)", folder, "Communication Efficiency")

        # -------------------------------------------------
        # 3. 对比 Sync Update (证明异步高吞吐)
        # -------------------------------------------------
        # 核心指标：Throughput
        folder = "3_Effect_of_Async"
        baseline = "Sync Update"
        print(f"Generating {folder}...")

        # 图1: 吞吐量对比
        plot_pair_line(df, baseline, 'tokens_per_s', "Throughput (Tokens/s)", folder, "System Throughput Analysis")

        # -------------------------------------------------
        # 4. 对比 Heuristic (证明算法优越性)
        # -------------------------------------------------
        # 核心指标：Latency CDF (长尾稳定性)
        folder = "4_Effect_of_Algorithm"
        baseline = "Heuristic Only"
        print(f"Generating {folder}...")

        # 图1: CDF 分布
        plot_pair_cdf(df, baseline, folder)
        # 图2: 延迟曲线
        plot_pair_line(df, baseline, 'step_time_ms', "Latency (ms)", folder, "Latency Stability")

        print(f"\n✅ All pairwise ablation plots saved to ./{ROOT_OUTPUT_DIR}")