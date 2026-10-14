"""
从维基百科（corpus/wiki_zh）与《现代汉语词典》（corpus/xdhycd）提取条目标题，
按字符表白名单过滤后合并去重，生成一份备用词表。

- 维基百科：每个文件为 JSON Lines（每行一个 JSON 对象），取其中的 title 字段。
- 《现代汉语词典》：每行形如【词条】释义，取【】内的词头。
- 只保留所有字符都在白名单内的词条，合并去重后排序写入 output/backup_lexicon.txt。

用法（在仓库根目录）：
    python build_backup_lexicon.py --wiki_dir ./corpus/wiki_zh --dict_dir ./corpus/xdhycd \
        --charset_path ./charset.txt --output_dir ./output
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from utils.logger import logger


def load_charset(charset_path: Path):
    """读取字符表白名单（每行一个字符）。"""
    with charset_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def iter_wiki_titles(wiki_dir: Path):
    """遍历 wiki_zh 目录下所有文件，每行一个 JSON，yield 其中的 title。"""
    files = sorted(p for p in wiki_dir.rglob("*") if p.is_file())
    for fp in tqdm(files, desc="Extracting wiki titles"):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"JSON 解析失败，跳过该行: {fp.name}: {line[:50]}...")
                    continue
                title = obj.get("title")
                if title:
                    yield title


def iter_dict_entries(dict_dir: Path):
    """遍历词典目录下所有 .txt，提取每行形如【词条】的词头。"""
    files = sorted(dict_dir.rglob("*.txt"))
    for fp in tqdm(files, desc="Extracting dict entries"):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("【"):
                    continue
                end = line.find("】")
                if end == -1:
                    continue
                entry = line[1:end].strip()
                if entry:
                    yield entry


def filter_by_charset(entries, charset):
    """只保留所有字符都在白名单内的词条。"""
    for entry in entries:
        if entry and all(ch in charset for ch in entry):
            yield entry


def main(wiki_dir: Path, dict_dir: Path, charset_path: Path, output_dir: Path, filename: str):
    if not wiki_dir.exists():
        logger.warning(f"维基百科目录不存在: {wiki_dir}")
    if not dict_dir.exists():
        logger.warning(f"词典目录不存在: {dict_dir}")

    charset = load_charset(charset_path)
    logger.info(f"Loaded charset from {charset_path} ({len(charset)} chars)")

    wiki_titles = list(iter_wiki_titles(wiki_dir))
    dict_entries = list(iter_dict_entries(dict_dir))
    logger.info(f"Raw titles: wiki={len(wiki_titles)}, dict={len(dict_entries)}")

    merged = wiki_titles + dict_entries
    filtered = list(filter_by_charset(merged, charset))
    logger.info(f"After charset filter: {len(filtered)} / {len(merged)}")

    unique = sorted(set(filtered))
    logger.info(f"Unique entries: {len(unique)}")

    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for word in unique:
            f.write(word + "\n")
    logger.info(f"Saved {len(unique)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从维基百科与《现代汉语词典》提取条目标题，生成备用词表。"
    )
    parser.add_argument("--wiki_dir", type=Path, default="./corpus/wiki_zh")
    parser.add_argument("--dict_dir", type=Path, default="./corpus/xdhycd")
    parser.add_argument("--charset_path", type=Path, default="./charset.txt")
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--filename", type=str, default="backup_lexicon.txt")
    args = parser.parse_args()
    main(args.wiki_dir, args.dict_dir, args.charset_path, args.output_dir, args.filename)
