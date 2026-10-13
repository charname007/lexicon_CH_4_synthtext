import tempfile
import unittest
from pathlib import Path

from plot_word_freq_loglog import downsample_points, plot_word_frequency_loglog


class PlotWordFrequencyLogLogTest(unittest.TestCase):
    def test_downsample_points_limits_svg_point_count(self):
        points = [
            {
                "word": f"词{i}",
                "rank": i,
                "frequency": 1000 - i,
                "log_rank": 0.0,
                "log_frequency": 0.0,
            }
            for i in range(1, 101)
        ]

        sampled_points = downsample_points(points, max_points=10)

        self.assertEqual(len(sampled_points), 10)
        self.assertEqual(sampled_points[0]["rank"], 1)
        self.assertEqual(sampled_points[-1]["rank"], 100)

    def test_writes_svg_log_log_plot_from_word_frequency_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "word_freq.csv"
            output_svg = temp_path / "word_freq_loglog.svg"

            input_csv.write_text(
                "\n".join(
                    [
                        "的,100,0.5",
                        "了,50,0.25",
                        "在,25,0.125",
                    ]
                ),
                encoding="utf-8",
            )

            plot_word_frequency_loglog(input_csv, output_svg)

            self.assertTrue(output_svg.exists())
            svg_text = output_svg.read_text(encoding="utf-8")
            self.assertIn("<svg", svg_text)
            self.assertIn("词频-排名双对数图", svg_text)
            self.assertEqual(svg_text.count("<circle "), 3)


if __name__ == "__main__":
    unittest.main()
