# ────────────────────────────────────────────────
#                  Unit Tests
# ────────────────────────────────────────────────
import unittest

from _temp_test import split_ja_phrases


class TestSplitJaPhrases(unittest.TestCase):
    def test_basic_splitting_mode_B(self):
        text = "私は昨日新宿で友達とラーメンを食べました。"
        result = split_ja_phrases(text, mode="B")
        expected = [
            "私",
            "は",
            "昨日",
            "新宿",
            "で",
            "友達",
            "と",
            "ラーメン",
            "を",
            "食べました",
            "。",
        ]
        self.assertEqual(result, expected)

    def test_mode_C_keeps_compounds(self):
        text = "私は国家公務員です。"
        result_c = split_ja_phrases(text, mode="C")
        result_b = split_ja_phrases(text, mode="B")
        self.assertIn("国家公務員", result_c)  # should stay together
        self.assertNotIn("国家公務員", "".join(result_b))  # split in B

    def test_empty_input(self):
        self.assertEqual(split_ja_phrases(""), [])
        self.assertEqual(split_ja_phrases("  　"), [])

    def test_mode_A_is_fine_grained(self):
        text = "食べさせられなかった"
        result = split_ja_phrases(text, mode="A")
        self.assertGreater(len(result), 4)  # should split a lot

    def test_join_with_space(self):
        text = "東京はとても面白い都市です"
        result = split_ja_phrases(text, mode="B", join_with_space=True)
        self.assertEqual(len(result), 1)
        self.assertIn(" ", result[0])

    def test_punctuation_is_kept_separate(self):
        text = "こんにちは！元気ですか？"
        result = split_ja_phrases(text)
        self.assertIn("！", result)
        self.assertIn("？", result)

    def test_filter_stopwords(self):
        text = "私は学生です。"
        result = split_ja_phrases(text, filter_stopwords=True)
        self.assertNotIn("は", result)
        self.assertNotIn("です", result)
        self.assertNotIn("。", result)


if __name__ == "__main__":
    unittest.main()
