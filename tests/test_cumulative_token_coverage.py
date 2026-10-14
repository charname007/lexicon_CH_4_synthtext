"""cumulative_token_coverage 的测试脚本。

用法：
    python tests/test_cumulative_token_coverage.py    # 打印 word_freq.csv 在阈值 0.995 下的结果，以及语料段落总数
    python -m unittest discover -s tests -p "test_cumulative_token_coverage.py"   # 跑单元测试
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
from build_lexicon import cumulative_token_coverage

DEFAULT_CSV = REPO_ROOT / "output" / "word_freq.csv"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus"


def load_word_freq(csv_path: Path) -> Counter:
    """读取 word_freq.csv（三列：word,freq,ratio，无表头），返回 word -> freq 计数。"""
    counter = Counter()
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            word, freq, _ratio = line.rstrip("\n").split(",")
            counter[word] = int(freq)
    return counter



def run_coverage(csv_path: Path = DEFAULT_CSV, threshold: float = 0.995) -> dict:
    """在真实词频上跑一次 cumulative_token_coverage，返回统计结果。

    save_frequency_to_file 被 mock 掉，避免覆盖已有的 output/word_freq_ctc.csv。
    """
    counter = load_word_freq(csv_path)
    with mock.patch.object(build_lexicon, "save_frequency_to_file"):
        kept_counter, kept_words = cumulative_token_coverage(counter, threshold=threshold)
    return {
        "total_words": len(counter),
        "kept_words": len(kept_words),
        "kept_ratio": len(kept_words) / len(counter),
        "coverage": sum(kept_counter.values()) / sum(counter.values()),
    }


class CumulativeTokenCoverageTest(unittest.TestCase):
    def test_stops_when_threshold_reached(self):
        counter = Counter({"a": 100, "b": 1, "c": 1})
        with mock.patch.object(build_lexicon, "save_frequency_to_file"):
            _, words = cumulative_token_coverage(counter, threshold=0.99)
        self.assertEqual(words, ["a", "b"])

    def test_single_word_covers_threshold(self):
        counter = Counter({"a": 100, "b": 1})
        with mock.patch.object(build_lexicon, "save_frequency_to_file"):
            _, words = cumulative_token_coverage(counter, threshold=0.5)
        self.assertEqual(words, ["a"])

    def test_empty_counter(self):
        with mock.patch.object(build_lexicon, "save_frequency_to_file"):
            kept, words = cumulative_token_coverage(Counter(), threshold=0.995)
        self.assertEqual(dict(kept), {})
        self.assertEqual(words, [])

    # def test_count_paragraphs_counts_blank_line_separated_blocks(self):
    #     with tempfile.TemporaryDirectory() as tmp:
    #         tmp_path = Path(tmp)
    #         (tmp_path / "a.txt").write_text(
    #             "第一段\n第二行\n\n第二段\n\n\n第三段\n", encoding="utf-8"
    #         )
    #         (tmp_path / "b.txt").write_text("只有一个段落\n", encoding="utf-8")
    #         self.assertEqual(count_paragraphs(tmp_path), 4)


if __name__ == "__main__":
    result = run_coverage(threshold=0.995)
    print(
        f"threshold=0.995: kept {result['kept_words']}/{result['total_words']} words"
        f" ({result['kept_ratio']:.4%}), coverage {result['coverage']:.6%}"
    )
