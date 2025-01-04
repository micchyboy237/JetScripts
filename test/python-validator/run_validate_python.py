from typing import Optional, List, TypedDict
import ast
import os
from jet.logger import logger


class CodeValidationError(TypedDict):
    message: str
    code: Optional[str]
    file: Optional[str]
    line: Optional[int]


class ValidatePythonUsageExamples:
    def __init__(self, code_source: str | list[str]):
        self.code_source = code_source

    def check_syntax_error(self) -> Optional[List[CodeValidationError]]:
        """
        Check if there are any syntax errors in the Python code.
        If there are multiple sources, it returns errors for each source.
        """
        errors: List[CodeValidationError] = []

        if isinstance(self.code_source, list):
            for source in self.code_source:
                errors_for_source = self._check_syntax_error_for_source(source)
                if errors_for_source:
                    errors.extend(errors_for_source)
        else:
            errors_for_source = self._check_syntax_error_for_source(
                self.code_source)
            if errors_for_source:
                errors.extend(errors_for_source)

        return errors if errors else None

    def _check_syntax_error_for_source(self, source: str) -> Optional[List[CodeValidationError]]:
        """
        Check syntax errors for a single source.
        """
        code = self._get_code_as_string(source)
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            error = CodeValidationError(
                message=f"SyntaxError: {e.msg}",
                file=os.path.realpath(
                    source) if os.path.exists(source) else None,
                line=e.lineno,
                code=code.splitlines()[e.lineno-1] if e.lineno else None
            )
            return [error]

    def _get_code_as_string(self, source: str) -> str:
        """
        Convert a file or string of code to a single string of code.
        """
        if isinstance(source, str):
            if os.path.exists(source):
                with open(source, 'r') as file:
                    return file.read()
            return source
        else:
            raise ValueError(
                "Each source in the list must be a file path (str) or a code string.")


def main():
    file_path = "mocks/sample_incorrect_syntax.py"
    code_example = """def my_function():
    print(Hello, World!')"""

    validator = ValidatePythonUsageExamples([code_example, file_path])

    logger.info("Code validation result:")
    errors = validator.check_syntax_error()
    if errors:
        for idx, error in enumerate(errors):
            logger.newline()
            logger.error(f"Syntax error #{idx + 1}:\nFile: {error['file']}\nLine: {
                         error['line']}\nError: {error['message']}\nCode: {error['code']}")
    else:
        logger.success("No unused code")


if __name__ == "__main__":
    main()
