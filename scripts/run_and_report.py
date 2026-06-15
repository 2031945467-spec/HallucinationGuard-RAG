import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipelines import BasePipeline, RAGCovePipeline, RAGPipeline, RAGVerifyPipeline
from src.retriever import DocumentRetriever


REFUSAL_KEYWORDS = [
    "信息不足",
    "未提及",
    "无法回答",
    "不能回答",
    "拒绝回答",
    "拒答",
    "缺乏确凿证据",
    "无法确定",
]
ERROR_KEYWORDS = ["调用失败", "APIConnectionError", "超时", "连接失败"]
METHOD_NAME_CN = {
    "Base_LLM": "基础大模型",
    "RAG": "RAG 基线",
    "RAG_Verify": "RAG + 事实验证",
    "RAG_CoVe": "RAG + CoVe",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s，。！？；：、“”‘’（）()《》【】\[\],.!?;:'\"`*_#>-]+", "", text)


def contains_keyword(answer: str, keywords: List[str]) -> bool:
    norm = normalize_text(answer)
    return any(normalize_text(keyword) in norm for keyword in keywords)


def is_refusal(answer: str) -> bool:
    return contains_keyword(answer, REFUSAL_KEYWORDS)


def is_error(answer: str) -> bool:
    return contains_keyword(answer, ERROR_KEYWORDS)


def char_ngram_f1(answer: str, ground_truth: str, n: int = 2) -> float:
    answer_norm = normalize_text(answer)
    truth_norm = normalize_text(ground_truth)
    if len(answer_norm) < n or len(truth_norm) < n:
        return float(answer_norm == truth_norm)
    answer_grams = {
        answer_norm[i : i + n] for i in range(len(answer_norm) - n + 1)
    }
    truth_grams = {
        truth_norm[i : i + n] for i in range(len(truth_norm) - n + 1)
    }
    overlap = len(answer_grams & truth_grams)
    if not overlap:
        return 0.0
    precision = overlap / len(answer_grams)
    recall = overlap / len(truth_grams)
    return 2 * precision * recall / (precision + recall)


def char_ngram_recall(answer: str, ground_truth: str, n: int = 2) -> float:
    answer_norm = normalize_text(answer)
    truth_norm = normalize_text(ground_truth)
    if len(answer_norm) < n or len(truth_norm) < n:
        return float(answer_norm == truth_norm)
    answer_grams = {
        answer_norm[i : i + n] for i in range(len(answer_norm) - n + 1)
    }
    truth_grams = {
        truth_norm[i : i + n] for i in range(len(truth_norm) - n + 1)
    }
    return len(answer_grams & truth_grams) / len(truth_grams)


def key_fact_coverage(answer: str, ground_truth: str) -> float:
    answer_norm = normalize_text(answer)
    truth_norm = normalize_text(ground_truth)
    latin_numbers = re.findall(
        r"[a-z]+[a-z0-9./+-]*|\d+(?:\.\d+)?", truth_norm
    )
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", truth_norm)
    stop = {
        "什么",
        "通常",
        "主要",
        "表示",
        "用于",
        "可以",
        "一个",
        "一种",
        "进行",
        "称为",
    }
    facts = latin_numbers + [
        chunk for chunk in chinese_chunks if chunk not in stop and len(chunk) >= 2
    ]
    if not facts:
        return 0.0
    return sum(fact in answer_norm for fact in facts) / len(facts)


def evaluate_answer(answer: str, ground_truth: str) -> Dict[str, float]:
    if is_error(answer):
        return {
            "correct": 0,
            "hallucination": 0,
            "answered": 0,
            "error": 1,
            "support_score": 0.0,
        }

    if normalize_text(ground_truth) == normalize_text("上下文中未提及"):
        correct = int(is_refusal(answer))
        return {
            "correct": correct,
            "hallucination": 0 if correct else 1,
            "answered": 0 if correct else 1,
            "error": 0,
            "support_score": float(correct),
        }

    if is_refusal(answer):
        return {
            "correct": 0,
            "hallucination": 0,
            "answered": 0,
            "error": 0,
            "support_score": 0.0,
        }

    answer_norm = normalize_text(answer)
    truth_norm = normalize_text(ground_truth)
    ngram_score = char_ngram_f1(answer, ground_truth)
    recall_score = char_ngram_recall(answer, ground_truth)
    fact_score = key_fact_coverage(answer, ground_truth)
    exact = truth_norm in answer_norm
    truth_numbers = set(re.findall(r"\d+(?:\.\d+)?", truth_norm))
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer_norm))
    number_consistent = not truth_numbers or truth_numbers.issubset(answer_numbers)
    correct = int(
        number_consistent
        and (exact or fact_score >= 0.8 or recall_score >= 0.30)
    )
    return {
        "correct": correct,
        "hallucination": 0 if correct else 1,
        "answered": 1,
        "error": 0,
        "support_score": round(
            max(float(exact), fact_score, ngram_score, recall_score), 4
        ),
    }


def get_ground_truth(item: dict) -> str:
    return item.get("ground_truth") or item.get("standard_answer", "")


def calculate_metrics(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [column[:-7] for column in raw_df.columns if column.endswith("_answer")]
    detail_records = []
    metric_records = []
    for method in methods:
        counts = {"correct": 0, "hallucination": 0, "answered": 0, "error": 0}
        for index, row in raw_df.iterrows():
            result = evaluate_answer(
                str(row[f"{method}_answer"]), str(row["ground_truth"])
            )
            for key in counts:
                counts[key] += int(result[key])
            detail_records.append(
                {
                    "row_id": int(index),
                    "method": method,
                    "method_cn": METHOD_NAME_CN.get(method, method),
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "answer": row[f"{method}_answer"],
                    **result,
                }
            )

        total = len(raw_df)
        answered = counts["answered"]
        metric_records.append(
            {
                "method": method,
                "method_cn": METHOD_NAME_CN.get(method, method),
                "total_samples": total,
                "correct_count": counts["correct"],
                "hallucination_count": counts["hallucination"],
                "answered_count": answered,
                "error_count": counts["error"],
                "accuracy": round(counts["correct"] / total, 4),
                "hallucination_rate": round(
                    counts["hallucination"] / answered if answered else 0.0, 4
                ),
                "coverage_rate": round(answered / total, 4),
                "error_rate": round(counts["error"] / total, 4),
                "avg_latency": round(
                    float(raw_df[f"{method}_latency"].mean()), 4
                ),
            }
        )

    metric_df = pd.DataFrame(metric_records).sort_values(
        by=["hallucination_rate", "accuracy", "coverage_rate", "avg_latency"],
        ascending=[True, False, False, True],
    )
    return metric_df, pd.DataFrame(detail_records)


def build_markdown_report(
    metric_df: pd.DataFrame, detail_df: pd.DataFrame, run_seconds: float
) -> str:
    lines = [
        "# 实验报告（自动生成）",
        "",
        "- 报告类型：基于已保存实验结果自动生成",
        f"- 样本数：{int(metric_df['total_samples'].max())}",
        f"- 总耗时：{run_seconds:.2f} 秒",
        "",
        "## 指标口径",
        "",
        "- 正确率：答案覆盖标准答案关键事实的比例。",
        "- 幻觉率：实际作答样本中，答案未覆盖标准事实且未拒答的比例。",
        "- 覆盖率：系统实际给出事实性答案的比例；拒答和调用失败不计入覆盖。",
        "- 失败率：模型服务连接、超时或调用失败的比例。",
        "",
        "## 方法对比",
        "",
        "| 方法 | 正确率 | 幻觉率 | 覆盖率 | 失败率 | 平均时延(秒) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metric_df.iterrows():
        lines.append(
            f"| {row['method_cn']} | {row['accuracy']:.2%} | "
            f"{row['hallucination_rate']:.2%} | {row['coverage_rate']:.2%} | "
            f"{row['error_rate']:.2%} | {row['avg_latency']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 逐题判定示例（前 12 条）",
            "",
            "| 题号 | 方法 | 正确 | 幻觉 | 作答 | 支持分 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in detail_df.head(12).iterrows():
        lines.append(
            f"| {int(row['row_id']) + 1} | {row['method_cn']} | "
            f"{int(row['correct'])} | {int(row['hallucination'])} | "
            f"{int(row['answered'])} | {row['support_score']:.3f} |"
        )
    return "\n".join(lines)


def write_evaluation(raw_df: pd.DataFrame, run_seconds: float) -> None:
    metric_df, detail_df = calculate_metrics(raw_df)
    metric_df.to_csv(
        results_dir / "experiment_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metric_df.to_json(
        results_dir / "experiment_metrics.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    detail_df.to_csv(
        results_dir / "experiment_eval_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (results_dir / "experiment_report.md").write_text(
        build_markdown_report(metric_df, detail_df, run_seconds), encoding="utf-8"
    )
    print(metric_df.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="不调用模型，仅重新评价已有 experiment_results.csv",
    )
    args = parser.parse_args()
    run_start = datetime.now()

    if args.evaluate_only:
        raw_df = pd.read_csv(results_dir / "experiment_results.csv")
        write_evaluation(raw_df, 0.0)
        return

    print("[1/5] 建立离线 TF-IDF 知识库索引", flush=True)
    retriever = DocumentRetriever()
    retriever.ingest_document(
        str(project_root / "data" / "raw_docs" / "knowledge.txt")
    )

    print("[2/5] 初始化四种实验管线", flush=True)
    pipelines = {
        "Base_LLM": BasePipeline(),
        "RAG": RAGPipeline(),
        "RAG_Verify": RAGVerifyPipeline(),
        "RAG_CoVe": RAGCovePipeline(),
    }
    test_data = json.loads(
        (project_root / "data" / "test_dataset.json").read_text(encoding="utf-8")
    )

    print("[3/5] 执行 100 条对比实验", flush=True)
    rows = []
    checkpoint_path = results_dir / "experiment_checkpoint.jsonl"
    for index, item in enumerate(test_data, 1):
        question = item["question"]
        row = {
            "id": item.get("id", f"Q{index}"),
            "category": item.get("category", ""),
            "question": question,
            "ground_truth": get_ground_truth(item),
        }
        for name, pipeline in pipelines.items():
            output = pipeline.run(question)
            row[f"{name}_answer"] = output["answer"]
            row[f"{name}_latency"] = round(output["latency"], 4)
            row[f"{name}_context"] = "\n---\n".join(output.get("context", []))
        rows.append(row)
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"   完成第 {index}/{len(test_data)} 题", flush=True)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(
        results_dir / "experiment_results.csv", index=False, encoding="utf-8-sig"
    )
    raw_df.to_json(
        results_dir / "experiment_results.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    checkpoint_path.unlink(missing_ok=True)

    print("[4/5] 计算正确率、幻觉率、覆盖率和失败率", flush=True)
    run_seconds = (datetime.now() - run_start).total_seconds()
    write_evaluation(raw_df, run_seconds)
    print("[5/5] 生成实验报告", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
