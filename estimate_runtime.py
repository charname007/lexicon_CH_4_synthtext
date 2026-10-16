"""Estimate the build_lexicon_v3 runtime by running its full pipeline on a corpus sample.

Example:
    python estimate_runtime.py --corpus_dir ./corpus --charset_path ./charset.txt
"""

import argparse
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from build_lexicon_v3 import build_lexicon
from utils.logger import logger


def select_sample_files(corpus_dir: Path, sample_ratio: float, seed: int) -> tuple[list[Path], int, int]:
    """Select a deterministic, source-stratified set of whole files by byte size."""
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio 必须在 (0, 1] 范围内")

    files = list(corpus_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"未找到语料 txt 文件: {corpus_dir}")

    total_bytes = sum(path.stat().st_size for path in files)
    target_bytes = max(1, int(total_bytes * sample_ratio))
    sources: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(corpus_dir)
        source = relative.parts[0] if len(relative.parts) > 1 else "."
        sources[source].append(path)

    rng = random.Random(seed)
    for source_files in sources.values():
        rng.shuffle(source_files)

    # Round-robin across top-level corpus sources so the sample is not dominated
    # by whichever source happens to contain the largest files.
    selected: list[Path] = []
    selected_bytes = 0
    source_queues = list(sources.values())
    while selected_bytes < target_bytes and source_queues:
        remaining_queues = []
        for queue in source_queues:
            if queue:
                path = queue.pop()
                selected.append(path)
                selected_bytes += path.stat().st_size
            if queue:
                remaining_queues.append(queue)
        source_queues = remaining_queues

    return selected, selected_bytes, total_bytes


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分钟"
    if minutes:
        return f"{minutes} 分钟 {seconds} 秒"
    return f"{seconds} 秒"


def main() -> None:
    parser = argparse.ArgumentParser(description="按语料样本估算 build_lexicon_v3 完整流程耗时")
    parser.add_argument("--corpus_dir", type=Path, default=Path("./corpus"))
    parser.add_argument("--charset_path", type=Path, default=Path("./charset.txt"))
    parser.add_argument("--sample_ratio", type=float, default=0.01,
                        help="抽样语料字节比例，默认 0.01（1%%）；会按完整文件抽样")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="每批段落数，需与正式运行参数一致")
    parser.add_argument("--files_per_group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    args = parser.parse_args()

    # Estimation logs should not append to the repository's regular log file.
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    corpus_dir = args.corpus_dir.resolve()
    selected, sample_bytes, total_bytes = select_sample_files(
        corpus_dir, args.sample_ratio, args.seed
    )
    actual_ratio = sample_bytes / total_bytes
    print(f"抽样文件数: {len(selected)}")
    print(f"语料总大小: {total_bytes / (1024 ** 3):.2f} GiB")
    print(f"抽样大小: {sample_bytes / (1024 ** 3):.2f} GiB ({actual_ratio:.2%})")
    print("开始在临时输出目录运行 v3 完整统计和指标流程…")

    with tempfile.TemporaryDirectory(prefix="lexicon_runtime_estimate_") as temp_dir:
        start = time.perf_counter()
        build_lexicon(
            corpus_dir=corpus_dir,
            charset_path=args.charset_path,
            output_dir=Path(temp_dir),
            segmentation_tool="hanlp",
            batch_size=args.batch_size,
            files_per_group=args.files_per_group,
            txt_files=selected,
        )
        sample_seconds = time.perf_counter() - start

    estimated_seconds = sample_seconds / actual_ratio
    print(f"样本完整流程耗时: {format_duration(sample_seconds)}")
    print(f"估算完整流程总耗时: {format_duration(estimated_seconds)}")
    print("估算按样本字节吞吐线性外推；正式运行参数、硬件和样本代表性会影响实际耗时。")


if __name__ == "__main__":
    main()
