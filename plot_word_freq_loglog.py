import argparse
import csv
import html
import math
from pathlib import Path

from utils.logger import logger


def load_word_frequencies(csv_path: Path):
    word_frequencies = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue

            word = row[0].strip()
            try:
                frequency = float(row[1])
            except ValueError:
                continue

            if not word or frequency <= 0:
                continue

            word_frequencies.append((word, frequency))

    if not word_frequencies:
        raise ValueError(f"No positive frequencies found in {csv_path}")

    return word_frequencies


def build_rank_frequency_points(word_frequencies):
    points = []
    for rank, (word, frequency) in enumerate(word_frequencies, start=1):
        points.append(
            {
                "word": word,
                "rank": rank,
                "frequency": frequency,
                "log_rank": math.log10(rank),
                "log_frequency": math.log10(frequency),
            }
        )
    return points


def fit_log_line(points):
    count = len(points)
    sum_x = sum(point["log_rank"] for point in points)
    sum_y = sum(point["log_frequency"] for point in points)
    sum_xy = sum(point["log_rank"] * point["log_frequency"] for point in points)
    sum_x2 = sum(point["log_rank"] ** 2 for point in points)

    denominator = count * sum_x2 - sum_x ** 2
    if denominator == 0:
        return 0.0, sum_y / count

    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return slope, intercept


def downsample_points(points, max_points=5000):
    if len(points) <= max_points:
        return points

    sampled_points = []
    last_index = len(points) - 1
    for sample_index in range(max_points):
        source_index = round(sample_index * last_index / (max_points - 1))
        sampled_points.append(points[source_index])

    return sampled_points


def map_to_canvas(value, value_min, value_max, canvas_min, canvas_max):
    if value_max == value_min:
        return (canvas_min + canvas_max) / 2

    ratio = (value - value_min) / (value_max - value_min)
    return canvas_min + ratio * (canvas_max - canvas_min)


def render_svg(points, output_path: Path, title: str):
    width = 1200
    height = 800
    left = 120
    right = 60
    top = 90
    bottom = 120
    plot_width = width - left - right
    plot_height = height - top - bottom

    sampled_points = downsample_points(points)

    x_values = [point["log_rank"] for point in points]
    y_values = [point["log_frequency"] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    slope, intercept = fit_log_line(points)

    circles = []
    for point in sampled_points:
        x = map_to_canvas(point["log_rank"], x_min, x_max, left, left + plot_width)
        y = map_to_canvas(point["log_frequency"], y_min, y_max, top + plot_height, top)
        circles.append(
            (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2563eb" opacity="0.72">'
                f"<title>{html.escape(point['word'])} | rank={point['rank']} | freq={point['frequency']:.0f}</title>"
                f"</circle>"
            )
        )

    line_start_y = slope * x_min + intercept
    line_end_y = slope * x_max + intercept
    line_x1 = map_to_canvas(x_min, x_min, x_max, left, left + plot_width)
    line_y1 = map_to_canvas(line_start_y, y_min, y_max, top + plot_height, top)
    line_x2 = map_to_canvas(x_max, x_min, x_max, left, left + plot_width)
    line_y2 = map_to_canvas(line_end_y, y_min, y_max, top + plot_height, top)

    grid_lines = []
    label_items = []
    tick_count = 6
    for index in range(tick_count):
        x_tick = x_min + (x_max - x_min) * index / (tick_count - 1) if tick_count > 1 else x_min
        y_tick = y_min + (y_max - y_min) * index / (tick_count - 1) if tick_count > 1 else y_min

        canvas_x = map_to_canvas(x_tick, x_min, x_max, left, left + plot_width)
        canvas_y = map_to_canvas(y_tick, y_min, y_max, top + plot_height, top)

        grid_lines.append(
            f'<line x1="{canvas_x:.2f}" y1="{top}" x2="{canvas_x:.2f}" y2="{top + plot_height}" '
            f'stroke="#e5e7eb" stroke-width="1" />'
        )
        grid_lines.append(
            f'<line x1="{left}" y1="{canvas_y:.2f}" x2="{left + plot_width}" y2="{canvas_y:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1" />'
        )

        label_items.append(
            f'<text x="{canvas_x:.2f}" y="{height - 70}" text-anchor="middle" fill="#374151" font-size="18">'
            f"{x_tick:.2f}</text>"
        )
        label_items.append(
            f'<text x="{left - 20}" y="{canvas_y + 6:.2f}" text-anchor="end" fill="#374151" font-size="18">'
            f"{y_tick:.2f}</text>"
        )

    subtitle = (
        "高频词和低频词跨度极大，普通坐标会压扁长尾分布；"
        "在双对数坐标下，Zipf 分布会更接近一条斜率约为 -1 的直线。"
    )
    fit_text = f"拟合直线: log10(f) = {slope:.3f} * log10(r) + {intercept:.3f}"
    sample_text = f"显示点数: {len(sampled_points)} / {len(points)}"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2:.0f}" y="42" text-anchor="middle" font-size="30" font-weight="700" fill="#111827">{html.escape(title)}</text>
<text x="{width / 2:.0f}" y="72" text-anchor="middle" font-size="17" fill="#4b5563">{html.escape(subtitle)}</text>
<g>
{''.join(grid_lines)}
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="2" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="2" />
<line x1="{line_x1:.2f}" y1="{line_y1:.2f}" x2="{line_x2:.2f}" y2="{line_y2:.2f}" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="8 6" />
{''.join(circles)}
{''.join(label_items)}
</g>
<text x="{width / 2:.0f}" y="{height - 28}" text-anchor="middle" font-size="22" fill="#111827">log10(排名 r)</text>
<text x="30" y="{height / 2:.0f}" text-anchor="middle" font-size="22" fill="#111827" transform="rotate(-90 30 {height / 2:.0f})">log10(词频 f)</text>
<text x="{left + 10}" y="{top + 26}" font-size="18" fill="#dc2626">{html.escape(fit_text)}</text>
<text x="{left + 10}" y="{top + 52}" font-size="18" fill="#2563eb">{html.escape(sample_text)}</text>
</svg>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def plot_word_frequency_loglog(
    input_csv: Path = Path("output/word_freq.csv"),
    output_svg: Path = Path("output/word_freq_loglog.svg"),
    title: str = "词频-排名双对数图",
):
    word_frequencies = load_word_frequencies(input_csv)
    points = build_rank_frequency_points(word_frequencies)
    render_svg(points, output_svg, title)
    logger.info(f"Saved log-log plot to {output_svg}")
    return output_svg


def main():
    parser = argparse.ArgumentParser(description="根据词频 CSV 绘制词频-排名双对数图（SVG）")
    parser.add_argument("--input_csv", type=Path, default=Path("output/word_freq.csv"), help="输入词频 CSV 路径")
    parser.add_argument("--output_svg", type=Path, default=Path("output/word_freq_loglog.svg"), help="输出 SVG 路径")
    parser.add_argument("--title", type=str, default="词频-排名双对数图", help="图标题")
    args = parser.parse_args()

    plot_word_frequency_loglog(args.input_csv, args.output_svg, args.title)


if __name__ == "__main__":
    main()
