import re
import unittest
from convert_notebooks_to_scripts import move_all_imports_on_top


def strip_lines(text: str) -> str:
    # Split the text into lines, strip spaces from each line
    stripped_text = '\n'.join(line.strip() for line in text.splitlines())
    # Replace consecutive newlines with a single newline
    return re.sub(r'\n+', '\n', stripped_text).strip()


class TestMoveImportsOnTop(unittest.TestCase):

    def test_single_line_import(self):
        code = """
        some_non_import_code()

        import os

        def example_function():
            pass
        """
        expected = """
        import os

        some_non_import_code()

        def example_function():
            pass
        """
        result = move_all_imports_on_top(code)
        self.assertEqual(strip_lines(result), strip_lines(expected))

    def test_multi_line_import(self):
        code = """
        some_non_import_code()

        from jet.logger import (
            logger,
            time_it
        )

        def example_function():
            pass
        """
        expected = """
        from jet.logger import (
            logger,
            time_it
        )

        some_non_import_code()

        def example_function():
            pass
        """
        result = move_all_imports_on_top(code)
        self.assertEqual(strip_lines(result), strip_lines(expected))

    def test_mixed_imports(self):
        code = """
        some_non_import_code()

        import os
        from jet.logger import logger
        from jet.logger import (
            logger,
            time_it
        )

        def example_function():
            pass
        """
        expected = """
        import os
        from jet.logger import logger
        from jet.logger import (
            logger,
            time_it
        )

        some_non_import_code()

        def example_function():
            pass
        """
        result = move_all_imports_on_top(code)
        self.assertEqual(strip_lines(result), strip_lines(expected))

    def test_no_imports(self):
        code = """
        some_non_import_code()

        def example_function():
            pass
        """
        expected = """
        some_non_import_code()

        def example_function():
            pass
        """
        result = move_all_imports_on_top(code)
        self.assertEqual(strip_lines(result), strip_lines(expected))

    def test_only_imports(self):
        code = """
        import os
        from jet.logger import logger
        from jet.logger import (
            logger,
            time_it
        )
        """
        expected = """
        import os
        from jet.logger import logger
        from jet.logger import (
            logger,
            time_it
        )
        """
        result = move_all_imports_on_top(code)
        self.assertEqual(strip_lines(result), strip_lines(expected))


if __name__ == "__main__":
    unittest.main()
