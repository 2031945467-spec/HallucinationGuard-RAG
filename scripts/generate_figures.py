from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

font_manager.fontManager.addfont(r"C:\Windows\Fonts\simhei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


def metric_chart():
    metrics = pd.read_csv(RESULTS / "experiment_metrics.csv")
    order = ["Base_LLM", "RAG", "RAG_Verify", "RAG_CoVe"]
    metrics = metrics.set_index("method").loc[order].reset_index()
    labels = ["Base LLM", "RAG", "RAG+Verify", "RAG+CoVe"]

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=180)
    x = range(len(metrics))
    width = 0.25
    colors = ["#D95F59", "#3D7EA6", "#2E8B77"]
    series = [
        ("hallucination_rate", "幻觉率（越低越好）", colors[0]),
        ("accuracy", "正确率", colors[1]),
        ("coverage_rate", "覆盖率", colors[2]),
    ]
    for offset, (column, label, color) in zip((-width, 0, width), series):
        bars = ax.bar(
            [value + offset for value in x],
            metrics[column] * 100,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("比例（%）")
    ax.set_title("四种方法的真实复现实验结果（100 条问题）", fontsize=16, pad=16)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    fig.savefig(FIGURES / "experiment_metrics.png", bbox_inches="tight")
    plt.close(fig)


def architecture_chart():
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=180)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.4, 1.8, 1.7, 1.2, "问题输入\n100 条测试集", "#E8EEF7"),
        (2.6, 1.8, 1.8, 1.2, "字符 n-gram\nTF-IDF 检索", "#D9EAF0"),
        (4.9, 1.8, 1.8, 1.2, "证据生成\n或直接回答", "#DDEEDC"),
        (7.2, 1.8, 1.8, 1.2, "事实验证\n置信度/一致性", "#F5E7C8"),
        (9.5, 1.8, 1.9, 1.2, "回答或拒答\n记录时延", "#F2D9D5"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)
    for x1, x2 in [(2.1, 2.6), (4.4, 4.9), (6.7, 7.2), (9.0, 9.5)]:
        ax.annotate(
            "",
            xy=(x2, 2.4),
            xytext=(x1, 2.4),
            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6),
        )
    ax.text(5.9, 4.25, "可复现的“检索—生成—验证—拒答”闭环", ha="center", fontsize=17)
    ax.text(
        5.9,
        0.65,
        "无 DeepSeek 密钥时自动使用本地 Ollama/Qwen；每题写入检查点，支持失败诊断",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "system_architecture.png", bbox_inches="tight")
    plt.close(fig)


def code_demo():
    code = """# 一条命令运行完整实验
python scripts/run_and_report.py

# 无 API 密钥时自动回退到本地模型
OLLAMA_MODEL=qwen2:0.5b

# 输出文件
results/experiment_results.csv
results/experiment_metrics.csv
results/experiment_eval_detail.csv
results/experiment_report.md

# 关键修复
1. TF-IDF 中文字符 n-gram 检索，替代随机哈希向量
2. 正确、幻觉、拒答、调用失败分别统计
3. 每题写入 checkpoint，避免长实验进度丢失"""
    fig, ax = plt.subplots(figsize=(11.5, 6.1), dpi=180)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.axis("off")
    ax.text(
        0.035,
        0.95,
        "真实运行入口与可复现输出",
        transform=ax.transAxes,
        va="top",
        color="#F8FAFC",
        fontsize=17,
        weight="bold",
    )
    ax.text(
        0.035,
        0.85,
        code,
        transform=ax.transAxes,
        va="top",
        color="#D8DEE9",
        fontsize=12.5,
        family="SimHei",
        linespacing=1.55,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "code_demo.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    metric_chart()
    architecture_chart()
    code_demo()
    print(FIGURES)
