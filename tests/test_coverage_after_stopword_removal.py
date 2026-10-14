"""测试：先删去停用词，再做累计覆盖率截断，结果会怎样。

对比两条路径：
  1. 不删停用词，直接 cumulative_token_coverage(0.995)
  2. 删去 hit_stopwords.txt 中出现的停用词后，再做 cumulative_token_coverage(0.995)

用法：
    python tests/test_coverage_after_stopword_removal.py    # 跑真实数据，结果写入 output/no_stopwords_coverage_report.txt
    python -m unittest discover -s tests -p "test_coverage_after_stopword_removal.py"   # 跑单元测试
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

WORD_FREQ_CSV = REPO_ROOT / "output" / "word_freq.csv"
STOPWORDS_PATH = REPO_ROOT / "hit_stopwords.txt"
COVERAGE_THRESHOLD = 0.995


def load_counter_csv(csv_path: Path) -> Counter:
    """读取 build_lexicon 生成的计数 CSV（word,count,ratio，无表头），返回 word -> count。"""
    counter = Counter()
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            word, count, _ratio = line.rstrip("\n").split(",")
            counter[word] = int(count)
    return counter


def load_stopwords(path: Path) -> set:
    """读取停用词表，strip 掉首尾空白并去重。"""
    words = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return words


def run_comparison(
    word_freq_csv: Path,
    stopwords_path: Path,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> dict:
    """对比「不删停用词」与「先删停用词」两种顺序下的覆盖率截断结果。"""
    word_freq = load_counter_csv(word_freq_csv)
    stopwords = load_stopwords(stopwords_path)

    original_total_tokens = sum(word_freq.values())

    # 删去词表中出现的停用词
    removed_words = {w for w in stopwords if w in word_freq}
    removed_tokens = sum(word_freq[w] for w in removed_words)
    no_stop = Counter({w: c for w, c in word_freq.items() if w not in removed_words})

    # 路径 1：不删停用词
    with mock.patch.object(build_lexicon, "save_frequency_to_file"):
        _, orig_kept = cumulative_token_coverage(word_freq, threshold=coverage_threshold)

    # 路径 2：先删停用词
    with mock.patch.object(build_lexicon, "save_frequency_to_file"):
        no_stop_kept_counter, no_stop_kept = cumulative_token_coverage(no_stop, threshold=coverage_threshold)

    no_stop_total_tokens = sum(no_stop.values())
    no_stop_kept_tokens = sum(no_stop_kept_counter.values())

    return {
        "total_words": len(word_freq),
        "total_tokens": original_total_tokens,
        "stopwords_removed": len(removed_words),
        "stopword_tokens": removed_tokens,
        "stopword_token_ratio": removed_tokens / original_total_tokens,
        "remaining_words": len(no_stop),
        "remaining_tokens": no_stop_total_tokens,
        "orig_kept_words": len(orig_kept),
        "no_stop_kept_words": len(no_stop_kept),
        "no_stop_kept_coverage_of_remaining": no_stop_kept_tokens / no_stop_total_tokens,
        "no_stop_kept_coverage_of_original": no_stop_kept_tokens / original_total_tokens,
        "no_stop_top": list(no_stop_kept)[:20],
        "no_stop_freq": no_stop,
    }


class CoverageAfterStopwordRemovalTest(unittest.TestCase):
    def test_removing_stopwords_changes_kept_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wf = tmp / "word_freq.csv"
            sw = tmp / "stopwords.txt"
            # 的 是停用词（高频），我/你/他 是内容词
            wf.write_text("的,1000,0.5\n我,600,0.3\n你,400,0.2\n他,1,0.0005\n", encoding="utf-8")
            sw.write_text("的\n", encoding="utf-8")

            result = run_comparison(wf, sw, coverage_threshold=0.995)

            self.assertEqual(result["stopwords_removed"], 1)
            self.assertEqual(result["remaining_words"], 3)  # 我/你/他
            # 去停用词后 total=1001，我(600)+你(400)=1000/1001=0.999 >= 0.995 即停，他 被截断
            self.assertEqual(result["no_stop_kept_words"], 2)
            self.assertNotIn("他", result["no_stop_top"])


if __name__ == "__main__":
    result = run_comparison(WORD_FREQ_CSV, STOPWORDS_PATH)

    print(f"total_tokens={result['total_tokens']}")
    print(
        f"stopwords_removed={result['stopwords_removed']} "
        f"stopword_tokens={result['stopword_tokens']} ({result['stopword_token_ratio']:.4%} of total)"
    )
    print(f"orig_coverage_kept_words={result['orig_kept_words']}")
    print(
        f"no_stop_coverage_kept_words={result['no_stop_kept_words']} "
        f"coverage_of_remaining={result['no_stop_kept_coverage_of_remaining']:.4%} "
        f"coverage_of_original={result['no_stop_kept_coverage_of_original']:.4%}"
    )

    report = REPO_ROOT / "output" / "no_stopwords_coverage_report.txt"
    lines = [
        f"original total tokens: {result['total_tokens']}",
        f"stopwords removed: {result['stopwords_removed']} "
        f"(tokens {result['stopword_tokens']}, {result['stopword_token_ratio']:.4%} of total)",
        f"original coverage kept words: {result['orig_kept_words']}",
        f"after stopword removal, coverage kept words: {result['no_stop_kept_words']}",
        f"  coverage of remaining tokens: {result['no_stop_kept_coverage_of_remaining']:.4%}",
        f"  coverage of original tokens: {result['no_stop_kept_coverage_of_original']:.4%}",
        "",
        "top 20 content words (after stopword removal):",
    ]
    for w in result["no_stop_top"]:
        lines.append(f"  {w}\t{result['no_stop_freq'][w]}")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}")
