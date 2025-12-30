import tempfile
import unittest
from pathlib import Path

from src.xnet import XNetConfig, XNetController


class TestXNetAlignment(unittest.TestCase):
    def test_load_gt_lines_parses_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = Path(tmp)
            (gt_dir / "001.gt.txt").write_text("A\n\n  B  \r\n\tC\n", encoding="utf8")

            controller = XNetController(XNetConfig(ground_truth_page_dir=gt_dir))
            lines = controller._load_gt_lines("001")
            self.assertEqual(lines, ["A", "  B", "\tC"])

    def test_align_gt_to_ocr_groups_prefers_small_merges(self) -> None:
        controller = XNetController(XNetConfig())
        gt = ["hello world", "goodbye"]
        ocr = ["hello", "world", "goodbye"]

        spans = controller._align_gt_to_ocr_groups(gt_lines=gt, ocr_lines=ocr, max_merge=3)
        self.assertEqual(spans, [(0, 2), (2, 3)])

    def test_line_similarity_is_whitespace_insensitive(self) -> None:
        controller = XNetController(XNetConfig())
        self.assertGreater(controller._line_similarity("A   B", "a b"), 0.99)

    def test_token_line_bboxes_group_by_line_num(self) -> None:
        controller = XNetController(XNetConfig())
        tokens = [
            {"text": "hello", "left": 10, "top": 5, "width": 20, "height": 10, "line_num": 1, "word_num": 1},
            {"text": "world", "left": 40, "top": 6, "width": 25, "height": 10, "line_num": 1, "word_num": 2},
            {"text": "bye", "left": 12, "top": 35, "width": 18, "height": 11, "line_num": 2, "word_num": 1},
            {"text": "now", "left": 34, "top": 34, "width": 16, "height": 11, "line_num": 2, "word_num": 2},
        ]

        bboxes = controller._token_line_bboxes_in_crop(tokens)
        self.assertEqual(len(bboxes), 2)
        # First line bbox should be above second.
        self.assertLess(bboxes[0][1], bboxes[1][1])
        # Basic bounds sanity.
        self.assertLess(bboxes[0][0], bboxes[0][2])
        self.assertLess(bboxes[0][1], bboxes[0][3])
