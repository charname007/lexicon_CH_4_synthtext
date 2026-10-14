"""测试：先做累计覆盖率截断（0.995），再按 rdf=0.01% 过滤，生成最终词表文件。

流程与 build_lexicon.filter 相反：这里先覆盖截断、后 rdf 过滤。

用法：
    python tests/test_ctc_then_rdf.py    # 跑真实数据并写出 output/lexicon_ctc0995_rdf0001.csv
    python -m unittest discover -s tests -p "test_ctc_then_rdf.py"   # 跑单元测试
"""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # build_lexicon 依赖仓库根的 utils 包

import build_lexicon
from build_lexicon import cumulative_token_coverage, save_word_list

WORD_FREQ_CSV = REPO_ROOT / "output" / "word_freq.csv"
DF_WORD_CSV = REPO_ROOT / "output" / "DF_word.csv"
# 语料段落总数，来自 tests/test_cumulative_token_coverage.py 的 count_paragraphs 结果
TOTAL_DOCS = 16605369

COVERAGE_THRESHOLD = 0.995
RDF_THRESHOLD = 0.0001  # 0.01%
OUTPUT_FILENAME = "lexicon_ctc0995_rdf0001.csv"


def load_counter_csv(csv_path: Path) -> Counter:
    """读取 build_lexicon 生成的计数 CSV（word,count,ratio，无表头），返回 word -> count。"""
    counter = Counter()
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            word, count, _ratio = line.rstrip("\n").split(",")
            counter[word] = int(count)
    return counter


def run_ctc_then_rdf(
    word_freq_csv: Path,
    df_csv: Path,
    total_docs: int,
    coverage_threshold: float = COVERAGE_THRESHOLD,
    rdf_threshold: float = RDF_THRESHOLD,
) -> dict:
    """先累计覆盖率截断，再按 rdf（DF/total_docs >= rdf_threshold）过滤。

    返回各阶段统计、每个保留词的 rdf 值、以及最终词频 Counter。
    """
    word_freq = load_counter_csv(word_freq_csv)
    df_counter = load_counter_csv(df_csv)

    # 第一步：累计覆盖率截断（mock 掉其写 word_freq_ctc.csv 的副作用，避免覆盖已有文件）
    with mock.patch.object(build_lexicon, "save_frequency_to_file"):
        ctc_counter, ctc_words = cumulative_token_coverage(word_freq, threshold=coverage_threshold)

    # 第二步：rdf 过滤，只对覆盖率保留的词计算 rdf = DF / total_docs
    rdf_kept = {}  # word -> rdf
    for word in ctc_words:
        df = df_counter.get(word, 0)
        rdf = df / total_docs
        if rdf >= rdf_threshold:
            rdf_kept[word] = rdf

    final_counter = Counter({w: ctc_counter[w] for w in rdf_kept})

    return {
        "total_words": len(word_freq),
        "ctc_words": len(ctc_words),
        "rdf_kept": len(rdf_kept),
        "final_words": len(final_counter),
        "rdf_values": rdf_kept,
        "final_counter": final_counter,
    }


class CtcThenRdfTest(unittest.TestCase):
    def test_coverage_then_rdf_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wf = tmp / "word_freq.csv"
            df = tmp / "df.csv"
            wf.write_text("a,100,0.66\nb,50,0.33\nc,1,0.01\n", encoding="utf-8")
            df.write_text("a,100,0.5\nb,5,0.025\nc,100,0.5\n", encoding="utf-8")

            result = run_ctc_then_rdf(
                wf, df, total_docs=1000, coverage_threshold=0.995, rdf_threshold=0.01
            )

            # 覆盖率阶段保留 a/b/c；rdf 阶段 b 的 rdf=0.005 < 0.01 被过滤掉
            self.assertEqual(set(result["final_counter"]), {"a", "c"})
            self.assertEqual(result["ctc_words"], 3)
            self.assertEqual(result["rdf_kept"], 2)


if __name__ == "__main__":
    result = run_ctc_then_rdf(
        WORD_FREQ_CSV, DF_WORD_CSV, TOTAL_DOCS, COVERAGE_THRESHOLD, RDF_THRESHOLD
    )

    min_df = RDF_THRESHOLD * TOTAL_DOCS
    print(f"total words             = {result['total_words']}")
    print(f"after coverage (0.995)  = {result['ctc_words']}")
    print(f"after rdf (>=0.01%, DF>={min_df:.0f}) = {result['rdf_kept']}")
    print(f"final words             = {result['final_words']}")

    save_word_list(
        result["final_counter"],
        output_dir=REPO_ROOT / "output",
        filename=OUTPUT_FILENAME,
    )
    print(f"wrote output/{OUTPUT_FILENAME}")
