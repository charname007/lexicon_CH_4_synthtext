"""对全量词频/文档频做阈值扫描，评测与《现代汉语常用词表（第2版）》的 Precision / Recall / F1。

在 output/word_freq.csv（词频）与 output/DF_word.csv（文档频）上，分别扫描：

- cumulative_token_coverage：按词频降序累加，直到累积覆盖率 >= threshold（复刻 build_lexicon.py 逻辑）。
- filter_by_df：保留 df/total_docs >= threshold 的词（复刻 build_lexicon.py 逻辑）。

并做 DF × CTC 二维组合网格（先 DF 后 CTC，与主流程 filter() 顺序一致）。

用法（仓库根目录）：
    python evaluate_thresholds.py \
        --word_freq output/word_freq.csv \
        --df_word output/DF_word.csv \
        --reference "corpus/现代汉语常用词表（第2版）.tsv"
"""

import argparse
import csv
from pathlib import Path

from compare_lexicon import evaluate, load_words

# 三张表的 CSV 列
FIELDNAMES = [
    "filter_type", "df_threshold", "ctc_threshold",
    "num_words", "tp", "ref_n", "precision", "recall", "f1",
]


def load_freq_csv(path: Path):
    """读取无表头 CSV（word,freq,ratio），返回 {word: freq}。"""
    freq = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0]:
                freq[row[0]] = float(row[1])
    return freq


def cumulative_coverage(word_freq: dict, threshold: float):
    """按词频降序累加，直到累积覆盖率 >= threshold，返回保留下来的词集合。"""
    total = sum(word_freq.values())
    if total <= 0:
        return set()
    kept = set()
    cumulative = 0
    for word, freq in sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True):
        kept.add(word)
        cumulative += freq
        if cumulative / total >= threshold:
            break
    return kept


def df_filter(df: dict, total_docs: int, threshold: float):
    """保留 df/total_docs >= threshold 的词集合。"""
    return {w for w, d in df.items() if d / total_docs >= threshold}


def to_row(filter_type, df_t, ctc_t, words, ref_set):
    """把一次评测结果整理成一行。"""
    tp, gen_n, ref_n, precision, recall, f1 = evaluate(words, ref_set)
    return {
        "filter_type": filter_type,
        "df_threshold": "" if df_t is None else f"{df_t:g}",
        "ctc_threshold": "" if ctc_t is None else f"{ctc_t:g}",
        "num_words": gen_n,
        "tp": tp,
        "ref_n": ref_n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sweep_ctc(word_freq: dict, ref_set: set, thresholds: list):
    rows = []
    for t in thresholds:
        words = cumulative_coverage(word_freq, t)
        rows.append(to_row("ctc", None, t, words, ref_set))
    return rows


def sweep_df(df: dict, total_docs: int, ref_set: set, thresholds: list):
    rows = []
    for t in thresholds:
        words = df_filter(df, total_docs, t)
        rows.append(to_row("df", t, None, words, ref_set))
    return rows


def sweep_grid(word_freq: dict, df: dict, total_docs: int, ref_set: set, df_thresholds: list, ctc_thresholds: list):
    rows = []
    for dt in df_thresholds:
        kept_by_df = df_filter(df, total_docs, dt)
        restricted = {w: word_freq[w] for w in kept_by_df if w in word_freq}
        for ct in ctc_thresholds:
            words = cumulative_coverage(restricted, ct)
            rows.append(to_row("grid", dt, ct, words, ref_set))
    return rows


def print_table(title, rows, df_key, ctc_key):
    print(f"\n=== {title} ===")
    header = f"{'df_threshold':>12} | {'ctc_threshold':>13} | {'词数':>10} | {'Precision':>9} | {'Recall':>9} | {'F1':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        df_v = r["df_threshold"] if r["df_threshold"] != "" else "-"
        ctc_v = r["ctc_threshold"] if r["ctc_threshold"] != "" else "-"
        print(f"{df_v:>12} | {ctc_v:>13} | {r['num_words']:>10} | {r['precision']:>9.2%} | {r['recall']:>9.2%} | {r['f1']:>9.2%}")


def main():
    parser = argparse.ArgumentParser(description="阈值扫描：词频覆盖率 / 文档频过滤 → Precision/Recall/F1")
    parser.add_argument("--word_freq", type=Path, default=Path("output/word_freq.csv"))
    parser.add_argument("--df_word", type=Path, default=Path("output/DF_word.csv"))
    parser.add_argument("--reference", type=Path, default=Path("corpus/现代汉语常用词表（第2版）.tsv"))
    parser.add_argument("--total_docs", type=int, default=16605169, help="总文档（段落）数，用于 filter_by_df 的 df/total_docs")
    parser.add_argument("--ctc_thresholds", type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    parser.add_argument("--df_thresholds", type=float, nargs="+",
                        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1])
    parser.add_argument("--output", type=Path, default=Path("output/threshold_sweep.csv"))
    args = parser.parse_args()

    word_freq = load_freq_csv(args.word_freq)
    df = load_freq_csv(args.df_word)
    ref_set, _ = load_words(args.reference, column=1, skip_header=True)

    print(f"word_freq 词数: {len(word_freq)}, df 词数: {len(df)}, 参考词数: {len(ref_set)}, total_docs: {args.total_docs}")

    rows = []
    rows += sweep_ctc(word_freq, ref_set, args.ctc_thresholds)
    rows += sweep_df(df, args.total_docs, ref_set, args.df_thresholds)
    rows += sweep_grid(word_freq, df, args.total_docs, ref_set, args.df_thresholds, args.ctc_thresholds)

    print_table("① cumulative_token_coverage 单独扫描", [r for r in rows if r["filter_type"] == "ctc"], "df", "ctc")
    print_table("② filter_by_df 单独扫描", [r for r in rows if r["filter_type"] == "df"], "df", "ctc")

    # 二维网格：按 df_threshold 分组打印
    grid_rows = [r for r in rows if r["filter_type"] == "grid"]
    print(f"\n=== ③ 二维组合网格（先 DF 后 CTC）F1 ===")
    ctc_labels = [f"{t:g}" for t in args.ctc_thresholds]
    header = f"{'df_threshold':>12} | " + " | ".join(f"{c:>9}" for c in ctc_labels)
    print(header)
    print("-" * len(header))
    for dt in args.df_thresholds:
        sub = [r for r in grid_rows if r["df_threshold"] == f"{dt:g}"]
        cells = " | ".join(f"{r['f1']:>9.2%}" for r in sub)
        print(f"{dt:>12g} | {cells}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已写入 {args.output} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
