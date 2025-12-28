import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
from matplotlib import font_manager

# ============================================================
# 配置：输出文件夹
# ============================================================
OUTPUT_DIR = "paper_figures"


# ============================================================
# 1. 顶会图表风格设置 (IEEE/ICWS Style)
# ============================================================
def set_pub_style():
    # 尝试设置 Times New Roman，如果系统没有则回退到 serif
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,  # 坐标轴标签字号
        'axes.titlesize': 18,  # 标题字号
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'legend.frameon': False,  # 去掉图例边框，显得更简洁
        'lines.linewidth': 2.5,  # 线条加粗
        'lines.markersize': 8,  # 标记点大小
        'figure.dpi': 300,  # 印刷级分辨率
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,  # 网格淡化
        'savefig.bbox': 'tight',  # 紧凑保存
    })


# 定义高对比度配色 (Colorblind-friendly + B&W friendly)
COLOR_MAP = {
    "Proposed (Full)": "#D62728",  # Brick Red (红色，最显眼)
    "w/o Hot/Cold": "#1F77B4",  # Muted Blue
    "Heuristic Only": "#FF7F0E",  # Safety Orange
    "Predictor Only": "#2CA02C",  # Cooked Asparagus Green
    "w/o NSGA-II": "#9467BD",  # Muted Purple
    "Sync Update": "#8C564B"  # Chestnut Brown
}

# 定义线型和标记，确保黑白打印也能区分
STYLE_MAP = {
    "Proposed (Full)": {"marker": "o", "linestyle": "-"},
    "w/o Hot/Cold": {"marker": "s", "linestyle": "--"},
    "Heuristic Only": {"marker": "^", "linestyle": "-."},
    "Predictor Only": {"marker": "v", "linestyle": ":"},
    "w/o NSGA-II": {"marker": "D", "linestyle": "--"},
    "Sync Update": {"marker": "X", "linestyle": "-."}
}


# ============================================================
# 2. 数据加载
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
    files = glob.glob("metrics_*.csv")
    if not files:
        print("[Error] 当前目录下未找到 metrics_*.csv 文件。")
        return None

    for f in files:
        try:
            df = pd.read_csv(f)
            mode_name = name_map.get(f, f.replace("metrics_", "").replace(".csv", ""))
            df['Method'] = mode_name

            # 预计算一些关键指标
            # 1. 标记是否发生了冷启动 (inv_cold_ms > 0)
            df['is_cold_start'] = df['inv_cold_ms'] > 0

            # 2. 平滑曲线 (Rolling Mean)
            df['latency_smooth'] = df['step_time_ms'].rolling(20).mean()
            df['loss_smooth'] = df['loss'].rolling(20).mean()

            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            pass

    return pd.concat(dfs, ignore_index=True) if dfs else None


# 辅助函数：保存图片到指定文件夹
def save_figure(filename_base):
    # 确保文件夹存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 保存 PDF (矢量图，论文首选)
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    plt.savefig(pdf_path, format='pdf')

    # 保存 PNG (预览用)
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    plt.savefig(png_path, format='png')

    print(f"Saved: {pdf_path} & {png_path}")


# ============================================================
# 3. 核心绘图逻辑
# ============================================================

def plot_latency_cdf(df):
    """ 图1: 延迟累积分布 (CDF) - 证明长尾延迟 (P99) 的优势 """
    plt.figure(figsize=(7, 5))

    methods = df['Method'].unique()
    # 排序：Proposed 放在最后绘制，防止被遮挡
    if "Proposed (Full)" in methods:
        methods = sorted([m for m in methods if m != "Proposed (Full)"]) + ["Proposed (Full)"]

    for method in methods:
        # 取出延迟数据并排序
        subset = df[df['Method'] == method]['step_time_ms'].dropna().sort_values()
        # 计算百分位 (0.0 ~ 1.0)
        y = np.arange(1, len(subset) + 1) / len(subset)

        c = COLOR_MAP.get(method, 'black')
        ls = STYLE_MAP.get(method, {}).get('linestyle', '-')

        plt.plot(subset, y, label=method, color=c, linestyle=ls, linewidth=2)

    plt.title("CDF of End-to-End Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Cumulative Probability")

    # 为了好看，截断极端的长尾 (显示 98% 的数据范围)
    limit = df['step_time_ms'].quantile(0.98)
    plt.xlim(0, limit)
    plt.ylim(0, 1.05)

    plt.legend(loc='lower right')
    save_figure("fig_paper_cdf")


def plot_cost_performance_tradeoff(df):
    """ 图2: 成本-性能权衡图 - 最强的综合证明 (左下角最优) """
    plt.figure(figsize=(7, 5))

    # 计算每个方法的平均延迟和平均单步成本
    summary = df.groupby('Method').agg({
        'step_time_ms': 'mean',
        'cost_usd_step': 'mean'
    }).reset_index()

    for _, row in summary.iterrows():
        method = row['Method']
        c = COLOR_MAP.get(method, 'black')
        m = STYLE_MAP.get(method, {}).get('marker', 'o')

        # 绘制散点，s=200 让点大一点
        plt.scatter(row['step_time_ms'], row['cost_usd_step'],
                    color=c, marker=m, s=200, label=method, edgecolors='k', zorder=10)

        # 添加文字标注，位置稍微上移一点
        plt.text(row['step_time_ms'], row['cost_usd_step'] * 1.005, method,
                 fontsize=10, ha='center', va='bottom', color='black', alpha=0.8)

    plt.title("Cost-Performance Trade-off")
    plt.xlabel("Average Latency (ms) [Lower is Better]")
    plt.ylabel("Avg Cost per Step ($) [Lower is Better]")

    # 绘制“更优区域”箭头 (指向左下角)
    xlim = plt.xlim()
    ylim = plt.ylim()
    # 计算箭头起始位置
    start_x = xlim[1] - (xlim[1] - xlim[0]) * 0.1
    start_y = ylim[1] - (ylim[1] - ylim[0]) * 0.1
    dx = -(xlim[1] - xlim[0]) * 0.15
    dy = -(ylim[1] - ylim[0]) * 0.15

    plt.arrow(start_x, start_y, dx, dy,
              head_width=(xlim[1] - xlim[0]) * 0.02, head_length=(ylim[1] - ylim[0]) * 0.03,
              fc='gray', ec='gray', alpha=0.5, width=(xlim[1] - xlim[0]) * 0.005)

    plt.text(start_x + dx, start_y + dy, "Better Region", fontsize=12, color='gray', ha='right', va='top')

    plt.grid(True, linestyle='--')
    save_figure("fig_paper_tradeoff")


def plot_cold_start_mitigation(df):
    """ 图3: 冷启动缓解效果 - 直接证明机制有效性 """
    plt.figure(figsize=(7, 5))

    # 重点对比 Proposed 和 w/o Hot/Cold
    target_methods = ["Proposed (Full)", "w/o Hot/Cold"]
    # 如果数据里有其他方法也可以加上
    existing_methods = [m for m in target_methods if m in df['Method'].unique()]

    if not existing_methods:
        existing_methods = df['Method'].unique()  # 如果没找到指定方法，就画所有的

    for method in existing_methods:
        subset = df[df['Method'] == method]
        # 使用 rolling mean (窗口50) 来观察冷启动开销的趋势
        smooth = subset['inv_cold_ms'].rolling(50).mean()

        c = COLOR_MAP.get(method, 'black')
        ls = STYLE_MAP.get(method, {}).get('linestyle', '-')

        plt.plot(subset['step'], smooth, label=method, color=c, linestyle=ls)

    plt.title("Cold Start Overhead Over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Avg Cold Start Latency (ms)")
    plt.legend()
    save_figure("fig_paper_coldstart")


def plot_training_convergence(df):
    """ 图4: 训练收敛曲线 - 证明优化没有损害精度 """
    plt.figure(figsize=(7, 5))

    methods = df['Method'].unique()
    if "Proposed (Full)" in methods:
        methods = sorted([m for m in methods if m != "Proposed (Full)"]) + ["Proposed (Full)"]

    for method in methods:
        subset = df[df['Method'] == method]
        c = COLOR_MAP.get(method, 'black')

        # Loss 往往震荡，需要平滑
        plt.plot(subset['step'], subset['loss_smooth'], label=method,
                 color=c, alpha=0.8, linewidth=1.5)

    plt.title("Training Convergence (Loss)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.ylim(bottom=0)
    plt.legend()
    save_figure("fig_paper_loss")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    set_pub_style()

    print(">>> Loading metrics data...")
    df = load_data()

    if df is not None:
        # 只取训练阶段数据进行分析
        df_train = df[df['phase'] == 'train']

        if df_train.empty:
            print("[Warning] Train phase data is empty!")
        else:
            print(f"Data loaded. Rows: {len(df_train)}")

            print(">>> Plotting CDF (Latency)...")
            plot_latency_cdf(df_train)

            print(">>> Plotting Trade-off (Cost vs Performance)...")
            plot_cost_performance_tradeoff(df_train)

            print(">>> Plotting Cold Start Analysis...")
            plot_cold_start_mitigation(df_train)

            print(">>> Plotting Convergence (Loss)...")
            plot_training_convergence(df_train)

            print(f"\n✅ 所有高清图表已保存至 '{OUTPUT_DIR}' 文件夹。")
            print("建议在论文中使用 .pdf 文件以获得最佳打印效果。")
    else:
        print("未找到数据，请检查当前目录下是否有 metrics_*.csv 文件。")