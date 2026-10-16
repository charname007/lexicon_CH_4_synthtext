"""离线阈值调参：读已落盘的「逐词分值表」，按阈值逐步求交，快速观察过滤结果。

只消费上一次跑批产出的 CSV —— 不做分词、不碰 sqlite，所以是秒级反馈。
各指标的「值」与阈值无关（指标值只是统计量的函数），因此换阈值无需重跑分词。

用法示例：
    python tune_filters.py --output_dir ./output --ctc 0.994 --rdf 1e-6
    python tune_filters.py --output_dir ./output --ctc 0.994 --rdf 1e-6 --dp 0.5 --universality 0.01

约定：
- 分值 CSV 为无表头 3 列「词, 值, 值/Σ值」；本脚本只用前两列，第三列忽略。
- ctc 特殊：word_freq.csv 存的是全量词频，阈值是「目标累积覆盖率」，
  需按词频降序累加到 >= 阈值来得到保留集合（word_freq_ctc.csv 内容与之相同）。
- 判据：ge 表示保留「值 >= 阈值」，le 表示保留「值 <= 阈值」。
- 仅在「指标已算过（分值表存在）」时有效；换分组粒度/口径仍需重跑统计。
"""

import argparse
import csv
from pathlib import Path

# 指标 -> (分值表文件名, 方向)；ctc 因阈值语义特殊，单独处理
METRIC_FILES = {
    "rdf": ("rdf.csv", "ge"),
    "cv": ("cv.csv", "le"),
    "dp": ("dp.csv", "le"),
    "dp_norm": ("dp_norm.csv", "le"),
    "universality": ("universality.csv", "ge"),
    "usage": ("usage.csv", "ge"),
}
# 应用顺序：ctc 先（截断总量），再逐个收窄
APPLY_ORDER = ["ctc", "rdf", "cv", "dp", "dp_norm", "universality", "usage"]
FREQ_FILE = "word_freq.csv"       # ctc 的输入；也用于展示覆盖率与样例


def read_two_col(path: Path) -> dict[str, float]:
    """读无表头 3 列 CSV，返回 {词: 值}（忽略第三列）。"""
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out[row[0]] = float(row[1])
    return out


def ctc_retained(freqs: dict[str, float], threshold: float) -> set[str]:
    """CTC：按词频降序累加，直到累积覆盖率 >= 阈值，返回保留集合。"""
    total = sum(freqs.values())
    if total <= 0:
        return set()
    kept: set[str] = set()
    cum = 0.0
    for w, f in sorted(freqs.items(), key=lambda kv: kv[1], reverse=True):
        kept.add(w)
        cum += f
        if cum / total >= threshold:
            break
    return kept


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="按阈值对已落盘的统计值表逐步求交（不跑分词，秒级反馈）"
    )
    p.add_argument("--output_dir", type=Path, default=Path("output"),
                   help="统计 CSV 所在目录（默认 ./output）")
    for name in APPLY_ORDER:
        if name == "ctc":
            help_text = "CTC 目标累积覆盖率（读 word_freq.csv）"
        else:
            fname, direction = METRIC_FILES[name]
            op = ">=" if direction == "ge" else "<="
            help_text = f"{fname}：保留 值 {op} 阈值"
        p.add_argument(f"--{name}", type=float, default=None, help=help_text)
    p.add_argument("--out", type=Path, default=None,
                   help="把最终词表逐行写入该 txt（可选）")
    p.add_argument("--samples", type=int, default=15,
                   help="每步展示的样例词数（默认 15）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    if not output_dir.is_dir():
        raise SystemExit(f"目录不存在: {output_dir}")

    existing = sorted(p.name for p in output_dir.glob("*.csv"))

    provided = [n for n in APPLY_ORDER if getattr(args, n) is not None]
    if not provided:
        print(f"未指定任何阈值。可用指标：{', '.join(APPLY_ORDER)}")
        print(f"目录 {output_dir} 现有 CSV：{', '.join(existing) or '(无)'}")
        return

    # 先校验所需文件是否齐备，避免为了一句报错白读一遍大 CSV
    missing: list[str] = []
    for name in provided:
        fname = FREQ_FILE if name == "ctc" else METRIC_FILES[name][0]
        if not (output_dir / fname).is_file():
            missing.append(f"{fname}（{name} 用）")
    if missing:
        raise SystemExit(
            f"缺少文件：{', '.join(missing)}；目录现有 CSV：{', '.join(existing) or '(无)'}"
        )

    freq_path = output_dir / FREQ_FILE
    freqs = read_two_col(freq_path) if freq_path.is_file() else {}
    total_freq = sum(freqs.values())

    print(f"目录: {output_dir}")
    if freqs:
        print(f"语料总量 total_words = {total_freq:.0f}（来自 {FREQ_FILE}）")
    else:
        print(f"未找到 {FREQ_FILE}，跳过总量展示与覆盖率统计")
    if "dp" in provided and "dp_norm" in provided:
        print("提示：dp 与 dp_norm 是同一指标的两种设置，同时求交通常无意义")
    print()

    current: set[str] | None = None
    for name in provided:
        thr = getattr(args, name)

        if name == "ctc":
            retained = ctc_retained(freqs, thr)
            criterion = f"累积覆盖率 >= {thr}"
        else:
            fname, direction = METRIC_FILES[name]
            table = read_two_col(output_dir / fname)
            if direction == "ge":
                retained = {w for w, v in table.items() if v >= thr}
                criterion = f"值 >= {thr}"
            else:
                retained = {w for w, v in table.items() if v <= thr}
                criterion = f"值 <= {thr}"

        before = current
        current = retained if current is None else (current & retained)

        line = f"[{name}] {criterion} → 本指标保留 {len(retained)}"
        if before is not None:
            dropped = before - current
            line += f"；累计 {len(current)}（本步淘汰 {len(dropped)}）"
        else:
            line += f"（起始集合）"
        print(line)

        # 展示本步淘汰的「最高频」词：最值得关注，避免误杀常用词
        if before is not None:
            newly_dropped = before - current
            if newly_dropped:
                top = sorted(newly_dropped, key=lambda w: freqs.get(w, 0.0), reverse=True)
                print(f"    淘汰样例(按词频降序): {' '.join(top[:args.samples])}")
        print()

    if current is None:
        return

    if total_freq:
        coverage = sum(freqs.get(w, 0.0) for w in current) / total_freq
        print(f"最终词表: {len(current)} 词；token 覆盖率 {coverage:.4%}")
    else:
        print(f"最终词表: {len(current)} 词")
    top_keep = sorted(current, key=lambda w: freqs.get(w, 0.0), reverse=True)
    print(f"保留样例(按词频降序): {' '.join(top_keep[:args.samples])}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for w in sorted(current):
                f.write(w + "\n")
        print(f"已写出词表: {args.out}")


if __name__ == "__main__":
    main()
