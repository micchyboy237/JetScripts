import ast
import os
from jet.logger import logger


class ValidatePythonUsageExamples:
    def __init__(self, code_source: str | list[str]):
        self.code_source = code_source

    def check_syntax_error(self):
        """
        Check if there are any syntax errors in the Python code.
        """
        code = self._get_code_as_string()

        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}"

    def _get_code_as_string(self):
        """
        Convert file or list of strings to a single string of code.
        """
        if isinstance(self.code_source, str):
            # If it's a file, read the content.
            if os.path.exists(self.code_source):
                with open(self.code_source, 'r') as file:
                    return file.read()
            return self.code_source
        elif isinstance(self.code_source, list):
            return '\n'.join(self.code_source)
        else:
            raise ValueError(
                "code_source must be a file path (str) or list of strings.")


def main():
    # Example 1: Validate code from a file
    file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/python-validator/sample_incorrect_syntax.py"
    validator_from_file = ValidatePythonUsageExamples(file_path)
    logger.info("File validation result:")
    validation_result = validator_from_file.check_syntax_error()
    if validation_result:
        logger.error(validation_result)
    else:
        logger.success(validation_result)

    # Example 2: Validate code from a list of strings
    code_example = """def my_function():
    print('Hello, World!')"""
    validator_from_list = ValidatePythonUsageExamples(code_example)
    logger.info("\nList of code validation result:")
    validation_result = validator_from_list.check_syntax_error()
    if validation_result:
        logger.error(validation_result)
    else:
        logger.success("No syntax errors")


if __name__ == "__main__":
    main()
