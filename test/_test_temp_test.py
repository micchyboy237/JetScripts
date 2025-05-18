import unittest
from _temp_test3 import extract_class_and_function_defs
import tempfile
import os
import jet.logger


class TestExtractHeadersAndDocstrings(unittest.TestCase):
    def write_temp_file(self, directory: str, filename: str, content: str) -> str:
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_extracts_docstrings_only(self):
        sample = '''\
def greet():
    """Say hello to the user"""
    print("Hello")

class Greeter:
    """Greeter class with simple behavior"""

    def say_hi(self):
        """Print a hi message"""
        print("Hi")
'''

        expected = [
            "def greet():\n    \"\"\"Say hello to the user\"\"\"",
            "class Greeter:\n    \"\"\"Greeter class with simple behavior\"\"\"",
            "    def say_hi(self):\n        \"\"\"Print a hi message\"\"\""
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = self.write_temp_file(
                tmpdirname, "docstrings_only.py", sample)
            result = extract_class_and_function_defs(tmpdirname)

            flat_results = result[path]
            self.assertEqual(flat_results, expected)


if __name__ == "__main__":
    unittest.main()
