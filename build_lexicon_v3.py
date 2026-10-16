"""build_lexicon 的 v3：新架构（word_freq 统计层 + metrics 指标层）下生成完整词表的最终流程。

与 build_lexicon.py 的区别：本文件是「最终流程」入口，统计与过滤全部交给
word_freq.count_word_frequency（一趟遍历、按需累计、分组倒排）与
metrics.apply_metrics（按 metrics 列表顺序求交），自身只做指标配置与落盘。

产物（写入 --output_dir）：
    word_freq.csv / char_freq.csv / DF_word.csv        全局统计
    rdf.csv / word_freq_ctc.csv                        lexicon 求交所用指标
    cv.csv / dp.csv / dp_norm.csv / universality.csv / usage.csv   报告用指标
    lexicon_v3.txt / lexicon_v3.csv                   最终词表

用法：
    python build_lexicon_v3.py --corpus_dir ./corpus --charset_path ./charset.txt --output_dir ./output
"""

import argparse
from collections import Counter
from pathlib import Path

from utils.logger import logger

# ===== 统计执行层（见 word_freq.py）=====
from word_freq import load_charset, count_word_frequency, open_group_store
# ===== 指标定义层（见 metrics.py）=====
from metrics import (
    DEFAULT_GROUP_KEY,
    required_instances,
    apply_metrics,
    RDFMetric,
    CTCMetric,
    CVMetric,
    GriesDPMetric,
    UniversalityMetric,
    UsageMetric,
)
# ===== 输出辅助（沿用 build_lexicon.py 中的实现）=====
from build_lexicon import save_frequency_to_file, save_word_list

# ===== 旧版遗留依赖：v3 改走 word_freq/metrics 后不再直接使用，按规则注释保留 =====
# import random
# import jieba
# import hanlp
# import torch
# from collections import defaultdict
# import csv
# import math
# import os
# import sqlite3
# import tempfile
# import regex
# from tqdm import tqdm


def build_metric_sets():
    """返回 (lexicon 求交用指标, 报告用指标)。

    - lexicon 求交：RDF(df/total_docs >= 1e-6) + CTC(累积覆盖率 >= 0.99)，
      复刻原 filter() 的「先 DF 后 CTC」；
    - 报告用：CV / Gries DP / 通用度 / 使用度，只计算并落 CSV，不参与求交。
    """
    lexicon_metrics = [RDFMetric(1e-6), CTCMetric(0.99)]
    report_metrics = [
        CVMetric(1.0),
        GriesDPMetric(1.0, group_key=DEFAULT_GROUP_KEY, normalize=False),
        UniversalityMetric(1.0, group_key=DEFAULT_GROUP_KEY, t=2),
        UsageMetric(1.0),
    ]
    return lexicon_metrics, report_metrics


def save_lexicon(lexicon: Counter, output_dir: Path, stem: str = "lexicon_v3") -> None:
    """写出最终词表：<stem>.txt（每行一词，频率降序）+ <stem>.csv（词, 归一化频率, 原始频率）。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{stem}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for word, _freq in lexicon.most_common():
            f.write(word + "\n")

    save_word_list(lexicon, output_dir=output_dir, filename=f"{stem}.csv")
    logger.info(f"Saved {len(lexicon)} words to {txt_path} and {output_dir / (stem + '.csv')}")


def build_lexicon(corpus_dir: Path, charset_path: Path, output_dir: Path,
                  segmentation_tool: str = "hanlp", batch_size: int = 64) -> Counter:
    """完整流程：统计 → 落各指标 CSV → 过滤求交 → 写最终词表；返回词表 Counter。"""
    charset = load_charset(charset_path)
    logger.info(f"Loaded charset from {charset_path}（{len(charset)} 字）")
    logger.info(f"Counted frequencies using {segmentation_tool}")

    lexicon_metrics, report_metrics = build_metric_sets()
    all_metrics = lexicon_metrics + report_metrics

    # 由指标声明推导这一趟需要维护哪些分组倒排表（只有 GROUP 统计才建表）
    active_gkeys = sorted({gk for _, gk in required_instances(all_metrics) if gk})
    logger.info(f"Active group keys: {active_gkeys or '无'}")

    # 临时 sqlite 生命周期必须覆盖「统计主循环 + 所有指标消费」
    with open_group_store(active_gkeys) as store:
        stats = count_word_frequency(corpus_dir, charset, segmentation_tool, all_metrics, store, batch_size)

        # 全局统计落 CSV
        save_frequency_to_file(stats["word_freq"], Path("word_freq.csv"), output_dir)
        save_frequency_to_file(stats["char_freq"], Path("char_freq.csv"), output_dir)
        save_frequency_to_file(stats["df_word"], Path("DF_word.csv"), output_dir)

        # lexicon 求交所用指标的中间产物
        save_frequency_to_file(lexicon_metrics[0].compute(stats), Path("rdf.csv"), output_dir)             # RDF
        save_frequency_to_file(lexicon_metrics[1].compute(stats), Path("word_freq_ctc.csv"), output_dir)   # CTC 输入词频

        # 报告用指标
        save_frequency_to_file(report_metrics[0].compute(stats), Path("cv.csv"), output_dir)               # CV
        save_frequency_to_file(report_metrics[1].compute(stats), Path("dp.csv"), output_dir)               # DP
        report_metrics[1].normalize = True                                                                 # DP_norm
        save_frequency_to_file(report_metrics[1].compute(stats), Path("dp_norm.csv"), output_dir)
        save_frequency_to_file(report_metrics[2].compute(stats), Path("universality.csv"), output_dir)     # 通用度
        save_frequency_to_file(report_metrics[3].compute(stats), Path("usage.csv"), output_dir)            # 使用度

        # 过滤：依次求交得到最终词表（必须在 with 内消费 group_tf）
        lexicon = apply_metrics(stats, lexicon_metrics)

    save_lexicon(lexicon, output_dir)
    return lexicon


def main():
    parser = argparse.ArgumentParser(description="build_lexicon v3：新架构下生成完整词表的最终流程")
    parser.add_argument("--corpus_dir", type=Path, default=Path("./corpus"))
    parser.add_argument("--charset_path", type=Path, default=Path("./charset.txt"))
    parser.add_argument("--output_dir", type=Path, default=Path("./output"))
    parser.add_argument("--segmentation_tool", "-s", type=str, default="hanlp",
                        choices=["hanlp"], help="分词工具（已弃用 jieba，仅支持 'hanlp'）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Segmentation batch size (paragraphs per batch, batch never crosses files)")

    args = parser.parse_args()
    build_lexicon(args.corpus_dir, args.charset_path, args.output_dir, args.segmentation_tool, args.batch_size)


if __name__ == "__main__":
    main()
