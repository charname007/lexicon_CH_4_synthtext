"""对比生成词表与参考常用词表，计算 Precision / Recall / F1。

用法示例：
    python core/compare_lexicon.py \
        --generated output/lexicon.csv \
        --reference "现代汉语常用词表（第2版）.tsv"

- generated：.csv 取第 1 列（跳过表头），.txt 每行取第一个空白分隔字段。
- reference：.tsv 取第 2 列「词语」（跳过表头），.txt / .csv 取第 1 列。


应该主要追求Recall覆盖率，因为现代汉语常用词表是11年左右的，而我的语料是19年左右，而且包含很多网络词汇，所以所以确实会有很多不属于常用词表的词
"""

import argparse
import csv
from pathlib import Path


def load_words(path: Path, column: int, skip_header: bool):
    """读取词表，返回 (set, list)，list 保持文件内出现顺序（已去重）。"""
    words = set()
    ordered = []

    def add(word: str):
        if word and word not in words:
            words.add(word)
            ordered.append(word)

    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    add(line.split()[0])  # 兼容「词\t频率」/「词 频率」
    else:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                if skip_header and i == 0:
                    continue
                if len(row) > column:
                    add(row[column].strip())

    return words, ordered


def evaluate(generated: set, reference: set):
    """返回 (tp, gen_n, ref_n, precision, recall, f1)。"""
    tp = len(generated & reference)
    gen_n = len(generated)
    ref_n = len(reference)
    precision = tp / gen_n if gen_n else 0.0
    recall = tp / ref_n if ref_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, gen_n, ref_n, precision, recall, f1


def dump_words(words, output_path: Path):
    """把词列表按行写入文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if words:
            f.write("\n".join(words) + "\n")


def main():
    parser = argparse.ArgumentParser(description="对比生成词表与参考常用词表，计算 Precision / Recall / F1")
    parser.add_argument("--generated", type=Path, default=Path("output/lexicon.txt"), help="生成的词表（.txt 或 .csv）")
    parser.add_argument("--reference", type=Path, default=Path("现代汉语常用词表（第2版）.tsv"), help="参考常用词表（.tsv / .txt / .csv）")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="差异文件输出目录（默认 output）")
    args = parser.parse_args()

    gen_suffix = args.generated.suffix.lower()
    gen_set, gen_order = load_words(args.generated, column=0, skip_header=(gen_suffix != ".txt"))

    ref_suffix = args.reference.suffix.lower()
    ref_column = 1 if ref_suffix == ".tsv" else 0
    ref_set, ref_order = load_words(args.reference, column=ref_column, skip_header=(ref_suffix != ".txt"))

    tp, gen_n, ref_n, precision, recall, f1 = evaluate(gen_set, ref_set)

    missing = [w for w in ref_order if w not in gen_set]  # 参考表有、生成表缺（按参考表频率序）
    extra = [w for w in gen_order if w not in ref_set]    # 生成表有、参考表无（按生成表频率序）

    print(f"参考词表: {args.reference}")
    print(f"生成词表: {args.generated}")
    print()
    print(f"生成词数: {gen_n}")
    print(f"参考词数: {ref_n}")
    print(f"交集词数: {tp}")
    print()
    print(f"Precision: {precision:.2%}  ({tp}/{gen_n})")
    print(f"Recall:    {recall:.2%}  ({tp}/{ref_n})")
    print(f"F1:        {f1:.2%}")
    print()

    missing_path = args.output_dir / "compare_missing.txt"
    extra_path = args.output_dir / "compare_extra.txt"
    dump_words(missing, missing_path)
    dump_words(extra, extra_path)
    print(f"参考表有而生成表缺（{len(missing)} 个）: {missing_path}")
    print(f"生成表有而参考表无（{len(extra)} 个）: {extra_path}")


if __name__ == "__main__":
    main()
