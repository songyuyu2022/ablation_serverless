import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = "figures_quality_proof"
# 定义要对比的方法（确保文件名能对应上）
TARGET_METHODS = [
    "Proposed (Full)",
    "w/o NSGA-II",
    "Heuristic Only",
    "w/o Hot/Cold",
    "Sync Update"
]
# 用于计算差异的基准方法 (Baseline)
# 通常用 "w/o Hot/Cold" 代表 Standard MoE (无优化)，或者 "Heuristic Only"
BASELINE_METHOD = "w/o Hot/Cold"


# ============================================================
# 0. 风格配置
# ============================================================
def set_style():
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
    })


COLOR_MAP = {
    "Proposed (Full)": "#D62728",  # Red
    "w/o Hot/Cold": "#1F77B4",  # Blue
    "Heuristic Only": "#FF7F0E",  # Orange
    "Predictor Only": "#2CA02C",  # Green
    "w/o NSGA-II": "#9467BD",  # Purple
    "Sync Update": "#7F7F7F"  # Grey
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ============================================================
# 1. 数据加载
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
            method = name_map.get(f, f.replace("metrics_", "").replace(".csv", ""))
            df['Method'] = method

            # 筛选训练数据
            if 'phase' in df.columns:
                df = df[df['phase'] == 'train']

            # 平滑
            for c in ['loss', 'acc_top1', 'acc_top5']:
                if c in df.columns:
                    df[f'{c}_smooth'] = df[c].rolling(window=50, min_periods=1).mean()

            dfs.append(df)
        except:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================
# 2. 生成表格 (CSV + LaTeX)
# ============================================================
def generate_quality_table(df):
    ensure_dir(OUTPUT_DIR)

    # 取最后 10% 的 step 作为收敛值
    summary_list = []

    for m in df['Method'].unique():
        sub = df[df['Method'] == m]
        if sub.empty: continue

        # 取尾部数据均值 (Stable performance)
        tail_len = max(10, int(len(sub) * 0.1))
        tail = sub.tail(tail_len)

        row = {
            "Method": m,
            "Final Loss": tail['loss'].mean(),
            "Final Top-1 Acc (%)": tail['acc_top1'].mean() * 100,
            "Final Top-5 Acc (%)": tail['acc_top5'].mean() * 100
        }
        summary_list.append(row)

    summary_df = pd.DataFrame(summary_list)

    # 排序：Proposed 第一，Baseline 第二，其他随后
    method_order = ["Proposed (Full)", BASELINE_METHOD] + [m for m in summary_df['Method'].unique() if
                                                           m not in ["Proposed (Full)", BASELINE_METHOD]]
    summary_df['Method'] = pd.Categorical(summary_df['Method'], categories=method_order, ordered=True)
    summary_df = summary_df.sort_values('Method').dropna()

    # 计算相对于 Baseline 的差异 (Diff)
    # 找到 Baseline 的数据
    base_row = summary_df[summary_df['Method'] == BASELINE_METHOD]

    if not base_row.empty:
        base_loss = base_row.iloc[0]['Final Loss']
        base_acc1 = base_row.iloc[0]['Final Top-1 Acc (%)']
        base_acc5 = base_row.iloc[0]['Final Top-5 Acc (%)']

        # 添加差异列
        summary_df['Loss Diff (%)'] = (summary_df['Final Loss'] - base_loss) / base_loss * 100
        summary_df['Acc@5 Diff (%)'] = (summary_df['Final Top-5 Acc (%)'] - base_acc5)
        # Acc Diff 直接用绝对百分点差异 (pp) 还是相对百分比? 通常 Acc 用绝对差 (e.g. +0.5%)
    else:
        print(f"[Warn] Baseline method '{BASELINE_METHOD}' not found in data.")
        summary_df['Loss Diff (%)'] = np.nan
        summary_df['Acc@5 Diff (%)'] = np.nan

    # 格式化
    print("\n" + "=" * 80)
    print(f"Quality Comparison Table (Baseline: {BASELINE_METHOD})")
    print("=" * 80)
    print(summary_df.to_string(index=False, float_format="%.4f"))
    print("=" * 80)

    # 1. 保存为 CSV
    csv_path = os.path.join(OUTPUT_DIR, "quality_table.csv")
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV table: {csv_path}")

    # 2. 保存为 LaTeX (直接贴进论文)
    tex_path = os.path.join(OUTPUT_DIR, "quality_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated LaTeX table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Quality Comparison (Non-degradation Proof)}\n")
        f.write("\\label{tab:quality_proof}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Final Loss & Top-1 Acc (\\%) & Top-5 Acc (\\%) & Diff vs Base (Acc@5) \\\\\n")
        f.write("\\midrule\n")
        for _, row in summary_df.iterrows():
            m = row['Method']
            loss = f"{row['Final Loss']:.4f}"
            acc1 = f"{row['Final Top-1 Acc (%)']:.2f}"
            acc5 = f"{row['Final Top-5 Acc (%)']:.2f}"

            # Diff 格式化
            if m == BASELINE_METHOD:
                diff = "-"
            else:
                d_val = row['Acc@5 Diff (%)']
                if pd.isna(d_val):
                    diff = "N/A"
                else:
                    diff = f"{d_val:+.2f} pp"  # pp = percentage points

            # 加粗 Proposed 行
            if "Proposed" in m:
                f.write(
                    f"\\textbf{{{m}}} & \\textbf{{{loss}}} & \\textbf{{{acc1}}} & \\textbf{{{acc5}}} & \\textbf{{{diff}}} \\\\\n")
            else:
                f.write(f"{m} & {loss} & {acc1} & {acc5} & {diff} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX table: {tex_path}")


# ============================================================
# 3. 绘图 (辅助证明)
# ============================================================
def plot_quality(df):
    ensure_dir(OUTPUT_DIR)

    # Top-5 Accuracy Curve
    plt.figure(figsize=(7, 5))

    methods = sorted(df['Method'].unique())
    if "Proposed (Full)" in methods:
        methods.remove("Proposed (Full)")
        methods.append("Proposed (Full)")

    for m in methods:
        sub = df[df['Method'] == m]
        if sub.empty: continue

        plt.plot(sub['step'], sub['acc_top5_smooth'],
                 label=m, color=COLOR_MAP.get(m, 'gray'),
                 linewidth=2.5 if "Proposed" in m else 1.5,
                 linestyle='-' if "Proposed" in m else '--',
                 alpha=1.0 if "Proposed" in m else 0.8)

    plt.xlabel("Training Step")
    plt.ylabel("Top-5 Accuracy")
    plt.title("Convergence Check: Top-5 Accuracy")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "quality_acc_curve.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    set_style()
    print(">>> Loading Data...")
    df = load_data()

    if df is not None:
        generate_quality_table(df)
        plot_quality(df)
        print("\n✅ Done. Check the 'figures_quality_proof' folder.")
    else:
        print("No metrics_*.csv data found.")