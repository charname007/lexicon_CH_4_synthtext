"""测试：累计覆盖率截断是否会过滤掉停用词（hit_stopwords.txt）。

结论先行：覆盖率截断保留的是「高频词」，而常见停用词（的/了/是/在…）恰恰是
词频最高的词，所以覆盖率截断不会过滤它们，反而把它们排在最前面。这里做对比验证。

用法：
    python tests/test_stopwords_vs_coverage.py    # 跑真实数据并打印对比
    python -m unittest discover -s tests -p "test_stopwords_vs_coverage.py"   # 跑单元测试
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


def compare_stopwords(
    word_freq_csv: Path,
    stopwords_path: Path,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> dict:
    """对比：停用词中有多少在词表里、多少被覆盖率截断保留/删除。"""
    word_freq = load_counter_csv(word_freq_csv)
    stopwords = load_stopwords(stopwords_path)

    with mock.patch.object(build_lexicon, "save_frequency_to_file"):
        _, ctc_words = cumulative_token_coverage(word_freq, threshold=coverage_threshold)

    rank = {w: i + 1 for i, w in enumerate(ctc_words)}  # 覆盖率截断后的排名（1 起）
    ctc_set = set(ctc_words)

    # 停用词中真正出现在词表里的，按词频降序
    in_full = sorted((w for w in stopwords if w in word_freq), key=lambda w: -word_freq[w])
    kept = [w for w in in_full if w in ctc_set]
    removed = [w for w in in_full if w not in ctc_set]

    return {
        "total_stopwords": len(stopwords),
        "in_full": in_full,
        "kept": kept,
        "removed": removed,
        "word_freq": word_freq,
        "rank": rank,
    }


class StopwordsVsCoverageTest(unittest.TestCase):
    def test_common_stopwords_kept_rare_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wf = tmp / "word_freq.csv"
            sw = tmp / "stopwords.txt"
            wf.write_text("的,100,0.5\n了,50,0.25\n我,49,0.245\n罕见停用词,1,0.005\n", encoding="utf-8")
            sw.write_text("的\n了\n我\n罕见停用词\n不在词表\n", encoding="utf-8")

            result = compare_stopwords(wf, sw, coverage_threshold=0.995)

            self.assertEqual(result["total_stopwords"], 5)
            self.assertEqual(len(result["in_full"]), 4)  # "不在词表" 不在词表里
            self.assertEqual(set(result["kept"]), {"的", "了", "我"})
            self.assertEqual(result["removed"], ["罕见停用词"])


if __name__ == "__main__":
    result = compare_stopwords(WORD_FREQ_CSV, STOPWORDS_PATH)

    print(f"total stopwords (unique) = {result['total_stopwords']}")
    print(f"in full lexicon          = {len(result['in_full'])}")
    print(f"kept by coverage (0.995) = {len(result['kept'])}")
    print(f"removed by coverage      = {len(result['removed'])}")
    print()

    print("kept stopwords (top 20, by coverage rank):")
    for w in result["kept"][:20]:
        print(f"  rank={result['rank'][w]:>8}  freq={result['word_freq'][w]:>9}  {w}")
    print()

    print("removed stopwords (all, sorted by freq):")
    for w in result["removed"]:
        print(f"  freq={result['word_freq'][w]:>9}  {w}")
