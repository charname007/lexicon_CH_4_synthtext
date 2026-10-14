"""build_lexicon 的 v2：自动扫阈值 → 按参考词表覆盖率打分 → 选最优临界点 → 生成词表。

主逻辑：
1. 复用已算好的 output/word_freq.csv 与 output/DF_word.csv；若不存在则回退
   build_lexicon.count_frequency 重新分词计数。
2. 扫 df_threshold × ctc_threshold 网格，对每个配置过滤出词表。
3. 与参考词表（现代汉语常用词表 / 现代汉语词典）比对，算 P/R/F1。
4. 选优：score() = Precision 达标(>= --precision_floor) 时的 Recall，否则 -inf；选 score 最高者。
   score() 是唯一的选优标准，想换（最高 F1、加权组合）只改这一个函数。
5. 保底：参考词表里在 word_freq 中出现过的词一律不丢（并集进最终词表，不参与选优）。


假设:对于分词工具错误的新词专有词，频率应该低于常用词，因此如果以追求常用词覆盖率作为标准，能过滤哪些低频不常用词

用法（仓库根目录）：
    python build_lexicon_v2.py --reference common --precision_floor 0.15
"""

import argparse
import csv
import re
from pathlib import Path

from tqdm import tqdm

from build_backup_lexicon import iter_dict_entries
from compare_lexicon import evaluate, load_words
from utils.logger import logger

# 与 build_lexicon.count_frequency 保持一致的句子切分（统计段落数时复用）
SENTENCE_SPLIT_PATTERN = r"(?<=[。！？；.!?])"


def load_freq_csv(path: Path):
    """读取无表头频率 CSV（word,freq,ratio），返回 {word: freq}。"""
    freq = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0]:
                freq[row[0]] = float(row[1])
    return freq


def load_word_list(path: Path):
    """读取一行一词的纯文本词表，返回词集合（用于合法性参照）。"""
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def count_documents(corpus_dir: Path) -> int:
    """统计文档（段落）总数，口径与 count_frequency 的 DF_sum 一致：空段落不计。"""
    files = sorted(p for p in corpus_dir.rglob("*.txt") if p.is_file())
    total = 0
    for txt_file in tqdm(files, desc="Counting documents"):
        with txt_file.open("r", encoding="utf-8") as f:
            para_lines = []
            for line in f:
                line = line.strip()
                if line:
                    para_lines.append(line)
                elif para_lines:
                    paragraph = "".join(para_lines)
                    para_lines = []
                    if any(re.split(SENTENCE_SPLIT_PATTERN, paragraph)):
                        total += 1
            if para_lines:
                paragraph = "".join(para_lines)
                if any(re.split(SENTENCE_SPLIT_PATTERN, paragraph)):
                    total += 1
    return total


def filter_by_df(df: dict, total_docs: int, threshold: float):
    """保留 df/total_docs >= threshold 的词集合。"""
    return {w for w, d in df.items() if d / total_docs >= threshold}


def cumulative_token_coverage(word_freq: dict, threshold: float):
    """按词频降序累加直到累积覆盖率 >= threshold，返回保留的词集合。"""
    total = sum(word_freq.values())
    ordered = sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True)
    return _cumulative_from_ordered(ordered, total, threshold)


def _cumulative_from_ordered(ordered, total: float, threshold: float):
    """从已按词频降序的 (word, freq) 列表累加，直到覆盖率 >= threshold，返回词集合。

    供扫描循环复用：同一 df 下 sort 一次，多个 ctc 阈值共用同一有序列表，避免重复排序。
    """
    if total <= 0:
        return set()
    kept = set()
    cumulative = 0
    for word, freq in ordered:
        kept.add(word)
        cumulative += freq
        if cumulative / total >= threshold:
            break
    return kept


def resolve_total_docs(explicit, path: Path, corpus_dir: Path) -> int:
    """解析 total_docs：显式值 > 记录文件 > 现场统计（并写回）。"""
    if explicit is not None:
        return explicit
    if path.exists():
        try:
            n = int(path.read_text(encoding="utf-8").strip())
            logger.info(f"从 {path} 读到 total_docs = {n}")
            return n
        except ValueError:
            logger.warning(f"{path} 内容无法解析，改为现场统计")
    n = count_documents(corpus_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(n), encoding="utf-8")
    logger.info(f"统计 total_docs = {n}，已写入 {path}")
    return n


def score(precision: float, recall: float, precision_floor: float) -> float:
    """选优评分：Precision 不低于下限时返回 Recall（越大越好），否则 -inf（不可用）。

    当前策略 = 「追 Recall + Precision 下限」。要换标准（最高 F1、加权组合等）只改这里。
    """
    if precision < precision_floor:
        return float("-inf")
    return recall


def load_reference(reference: str, reference_path: Path, dict_dir: Path) -> set:
    """加载参考词表：common = 现代汉语常用词表；dict = 现代汉语词典词头。"""
    if reference == "common":
        ref_set, _ = load_words(reference_path, column=1, skip_header=True)
    elif reference == "dict":
        ref_set = set(iter_dict_entries(dict_dir))
    else:
        raise ValueError(f"未知参考类型: {reference}")
    return ref_set


def main():
    parser = argparse.ArgumentParser(description="build_lexicon v2：扫阈值 → score 选优 → 生成词表")
    parser.add_argument("--corpus_dir", type=Path, default=Path("./corpus"))
    parser.add_argument("--charset_path", type=Path, default=Path("./charset.txt"))
    parser.add_argument("--segmentation_tool", "-s", type=str, default="jieba", choices=["jieba", "hanlp"])
    parser.add_argument("--word_freq", type=Path, default=Path("output/word_freq.csv"))
    parser.add_argument("--df_word", type=Path, default=Path("output/DF_word.csv"))
    parser.add_argument("--total_docs", type=int, default=None,
                        help="总文档（段落）数；不传则读 output/total_docs.txt，再没有则统计段落数")
    parser.add_argument("--reference", type=str, default="common", choices=["common", "dict"],
                        help="打分参考：common=现代汉语常用词表，dict=现代汉语词典")
    parser.add_argument("--reference_path", type=Path, default=Path("reference/现代汉语常用词表（第2版）.tsv"))
    parser.add_argument("--dict_dir", type=Path, default=Path("reference/xdhycd"))
    parser.add_argument("--validity_path", type=Path, default=Path("output/lexicon（non-frequency）.txt"),
                        help="合法性参照（词典+维基标题，一行一词），只用于报告合法性 precision，不参与选优/保底")
    parser.add_argument("--precision_floor", type=float, default=0.15,
                        help="score 的 Precision 下限：Recall 只在 P 达标时才有意义")
    parser.add_argument("--df_thresholds", type=float, nargs="+",
                        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1])
    parser.add_argument("--ctc_thresholds", type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.985,
                                 0.99, 0.992, 0.994, 0.996, 0.998, 0.999])
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    total_docs_path = args.output_dir / "total_docs.txt"

    # 1. 词频/文档频：有文件则复用，否则重新分词计数
    if args.word_freq.exists() and args.df_word.exists():
        word_freq = load_freq_csv(args.word_freq)
        df = load_freq_csv(args.df_word)
        total_docs = resolve_total_docs(args.total_docs, total_docs_path, args.corpus_dir)
    else:
        logger.info(f"频率文件不存在，重新分词计数（{args.segmentation_tool}）...")
        from build_lexicon import count_frequency, load_charset  # 懒加载，避免快路径引入 jieba/hanlp/torch
        charset = load_charset(args.charset_path)
        word_freq, _words_sum, _char_freq, _char_sum, df, total_docs, _word_sum_sq, dp_db_path = count_frequency(
            args.corpus_dir, charset, args.segmentation_tool
        )
        Path(dp_db_path).unlink(missing_ok=True)
        total_docs_path.parent.mkdir(parents=True, exist_ok=True)
        total_docs_path.write_text(str(total_docs), encoding="utf-8")

    # 2. 参考词表
    ref_set = load_reference(args.reference, args.reference_path, args.dict_dir)
    logger.info(f"参考[{args.reference}] 词数: {len(ref_set)}, word_freq 词数: {len(word_freq)}, total_docs: {total_docs}")

    # 2b. 合法性参照（词典+维基标题），只用于报告合法性 precision，不参与选优/保底
    validity_set = set()
    if args.validity_path.exists():
        validity_set = load_word_list(args.validity_path)
        logger.info(f"合法性参照（词典+维基）词数: {len(validity_set)}")
    else:
        logger.warning(f"合法性参照文件不存在，跳过: {args.validity_path}")

    # 3. 扫阈值网格，记录每个配置的指标（每个 df 只 sort 一次，所有 ctc 复用有序列表）
    results = []
    for dt in args.df_thresholds:
        kept_by_df = filter_by_df(df, total_docs, dt)
        restricted = {w: word_freq[w] for w in kept_by_df if w in word_freq}
        ordered = sorted(restricted.items(), key=lambda kv: kv[1], reverse=True)
        restricted_total = sum(restricted.values())
        for ct in args.ctc_thresholds:
            words = _cumulative_from_ordered(ordered, restricted_total, ct)
            _tp, gen_n, _ref_n, precision, recall, f1 = evaluate(words, ref_set)
            results.append({"df": dt, "ctc": ct, "words": gen_n, "p": precision, "r": recall, "f1": f1})

    # 4. 选优：score() 打分（recall 优先 + precision 下限），全不达标则退化为取 precision 最高者
    feasible = [r for r in results if score(r["p"], r["r"], args.precision_floor) != float("-inf")]
    if feasible:
        chosen = max(feasible, key=lambda r: score(r["p"], r["r"], args.precision_floor))
    else:
        logger.warning(f"没有配置满足 Precision >= {args.precision_floor}，退化为取 Precision 最高者")
        chosen = max(results, key=lambda r: r["p"])

    # 打印扫描结果
    print(f"\n参考[{args.reference}] 词数={len(ref_set)}, precision_floor={args.precision_floor}")
    print(f"{'df':>8} | {'ctc':>7} | {'words':>8} | {'P':>7} | {'R':>7} | {'F1':>7}")
    print("-" * 58)
    for rec in results:
        mark = "  <- 选中" if (rec["df"] == chosen["df"] and rec["ctc"] == chosen["ctc"]) else ""
        print(f"{rec['df']:>8g} | {rec['ctc']:>7g} | {rec['words']:>8} | {rec['p']:>7.2%} | {rec['r']:>7.2%} | {rec['f1']:>7.2%}{mark}")

    logger.info(
        f"选中临界点: df={chosen['df']:g}, ctc={chosen['ctc']:g}, 词数={chosen['words']}, "
        f"P={chosen['p']:.2%}, R={chosen['r']:.2%}, F1={chosen['f1']:.2%}"
    )

    # 5. 重新过滤出选中配置的词表（不缓存每个配置的词集，省内存）
    kept_by_df = filter_by_df(df, total_docs, chosen["df"])
    restricted = {w: word_freq[w] for w in kept_by_df if w in word_freq}
    chosen_words = cumulative_token_coverage(restricted, chosen["ctc"])

    # 6. 保底：参考词表里在 word_freq 中出现过的词，一律不丢（只作用在最终输出，不参与选优）
    protected = {w for w in ref_set if w in word_freq}
    final_words = chosen_words | protected
    logger.info(
        f"过滤词数: {len(chosen_words)}, 保底补回: {len(final_words) - len(chosen_words)}, "
        f"最终: {len(final_words)}, 最终 Recall: {len(final_words & ref_set) / len(ref_set):.2%}"
    )
    if validity_set:
        logger.info(
            f"合法性 precision（vs 词典+维基）: {len(final_words & validity_set)}/{len(final_words)} "
            f"= {len(final_words & validity_set) / len(final_words):.2%}"
        )

    kept_freq = {w: word_freq[w] for w in final_words}
    ordered = sorted(kept_freq.items(), key=lambda kv: kv[1], reverse=True)

    txt_path = args.output_dir / "lexicon_v2.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for word, _ in ordered:
            f.write(word + "\n")

    total_freq = sum(kept_freq.values())
    csv_path = args.output_dir / "lexicon_v2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "normalized_frequency", "raw_frequency"])
        for word, freq in ordered:
            writer.writerow([word, freq / total_freq if total_freq else 0.0, freq])

    logger.info(f"Saved {len(kept_freq)} words to {txt_path} and {csv_path}")


if __name__ == "__main__":
    main()
