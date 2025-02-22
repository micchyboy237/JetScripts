import unittest
from execute_extract_job_entities import merge_dot_prefixed_words


class TestMergeDotPrefixedWords(unittest.TestCase):
    def test_basic_merging(self):
        self.assertEqual(merge_dot_prefixed_words(
            "hello .world"), "hello.world")
        self.assertEqual(merge_dot_prefixed_words(
            'hello. world'), "hello.world")
        self.assertEqual(merge_dot_prefixed_words(
            "foo .bar .baz"), "foo.bar.baz")
        self.assertEqual(merge_dot_prefixed_words(
            "this is .a test"), "this is.a test")

    def test_no_merge_needed(self):
        self.assertEqual(merge_dot_prefixed_words(
            "this is normal"), "this is normal")
        self.assertEqual(merge_dot_prefixed_words(
            "another test case"), "another test case")

    def test_multiple_dots(self):
        self.assertEqual(merge_dot_prefixed_words(
            "one .two .three .four"), "one.two.three.four")
        self.assertEqual(merge_dot_prefixed_words("a .b .c d .e"), "a.b.c d.e")

    def test_empty_input(self):
        self.assertEqual(merge_dot_prefixed_words(""), "")

    def test_only_dot_prefixed_words(self):
        self.assertEqual(merge_dot_prefixed_words(".a .b .c"),
                         ".a .b .c")  # No previous word to merge with
        self.assertEqual(merge_dot_prefixed_words(".alone"), ".alone")

    def test_whitespace_variants(self):
        self.assertEqual(merge_dot_prefixed_words(
            "  leading space"), "leading space")
        self.assertEqual(merge_dot_prefixed_words(
            "trailing space  "), "trailing space")
        self.assertEqual(merge_dot_prefixed_words(
            "  spaces   between words  "), "spaces between words")
        # Extra spaces should be normalized
        self.assertEqual(merge_dot_prefixed_words("word  .dot"), "word.dot")


if __name__ == "__main__":
    unittest.main()
