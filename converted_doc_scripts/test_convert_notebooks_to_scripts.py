import unittest
from copy import deepcopy
from convert_notebooks_to_scripts import merge_consecutive_same_type


class TestMergeConsecutiveSameType(unittest.TestCase):
    def test_merge_basic(self):
        source = [
            {"type": "text", "code": "Hello"},
            {"type": "text", "code": "World"},
            {"type": "code", "code": "print('Hello')"},
            {"type": "code", "code": "print('World')"},
            {"type": "text", "code": "Again"}
        ]
        expected = [
            {"type": "text", "code": "Hello\n\nWorld"},
            {"type": "code", "code": "print('Hello')\n\nprint('World')"},
            {"type": "text", "code": "Again"}
        ]
        self.assertEqual(merge_consecutive_same_type(source), expected)

    def test_no_merge_needed(self):
        source = [
            {"type": "text", "code": "Hello"},
            {"type": "code", "code": "print('Hello')"},
            {"type": "text", "code": "World"},
        ]
        self.assertEqual(merge_consecutive_same_type(source), source)

    def test_empty_list(self):
        self.assertEqual(merge_consecutive_same_type([]), [])

    def test_single_element(self):
        source = [{"type": "text", "code": "Only one"}]
        self.assertEqual(merge_consecutive_same_type(source), source)

    def test_large_merge(self):
        source = [{"type": "code", "code": f"Line {i}"} for i in range(5)]
        expected = [{"type": "code", "code": "\n\n".join(
            f"Line {i}" for i in range(5))}]
        self.assertEqual(merge_consecutive_same_type(source), expected)

    def test_alternating_types(self):
        source = [
            {"type": "text", "code": "A"},
            {"type": "code", "code": "B"},
            {"type": "text", "code": "C"},
            {"type": "code", "code": "D"},
        ]
        self.assertEqual(merge_consecutive_same_type(source), source)


if __name__ == "__main__":
    unittest.main()
