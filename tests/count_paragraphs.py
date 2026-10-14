
## 统计语料中段落总数的单元测试
## 3783 个文件
## 16,605,369（约 1660 万段）

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock
import build_lexicon
import argparse

def count_paragraphs(corpus_dir: Path) -> int:
    """统计语料中段落（para）总数。

    划分规则与 build_lexicon.load_corpus 保持一致：只统计 *.txt 文件，
    按空行分割，连续的非空行（strip 后非空）合并为一个段落。
    """
    corpus_dir = Path(corpus_dir)
    total = 0
    for txt_file in sorted(corpus_dir.rglob("*.txt")):
        with txt_file.open("r", encoding="utf-8") as f:
            para_lines = []
            for line in f:
                line = line.strip()
                if line:
                    para_lines.append(line)
                elif para_lines:
                    total += 1
                    para_lines = []
            if para_lines:
                total += 1
    return total


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="count_paragraphs",
        description="Count paragraphs in a corpus directory.",
    )
    parser.add_argument("corpus_dir", type=Path, help="Path to the corpus directory.")
    args = parser.parse_args()

    total_paragraphs = count_paragraphs(args.corpus_dir)
    print(f"Total paragraphs in {args.corpus_dir}: {total_paragraphs}")