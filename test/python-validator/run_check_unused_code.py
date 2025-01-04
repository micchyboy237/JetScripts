import json
import os
import subprocess
from typing import List, Optional, TypedDict, Union, Literal
from jet.executor.command import run_command
from jet.logger import logger


class CodeValidationError(TypedDict):
    message: str
    code: Optional[str]
    file: Optional[str]
    line: Optional[int]


def check_unused_code(
    paths: Union[str, List[str]],
    exclude: Optional[List[str]] = None,
    ignore_decorators: Optional[List[str]] = None,
    ignore_names: Optional[List[str]] = None,
    make_whitelist: bool = False,
    min_confidence: Optional[int] = None,
    sort_by_size: bool = False,
    config: Optional[str] = None,
    verbose: bool = False
) -> Optional[list[CodeValidationError]]:
    """
    Run Vulture to check for unused code in Python files or directories.

    Args:
        paths (Union[str, List[str]]): File or directory paths to analyze.
        exclude (Optional[List[str]]): Paths to exclude from analysis.
        ignore_decorators (Optional[List[str]]): Decorators to ignore.
        ignore_names (Optional[List[str]]): Names to ignore.
        make_whitelist (bool): If True, generate a whitelist of unused code.
        min_confidence (Optional[int]): Minimum confidence for reporting unused code (0-100).
        sort_by_size (bool): If True, sort results by size of unused functions/classes.
        config (Optional[str]): Path to a pyproject.toml config file.
        verbose (bool): If True, enable verbose output.
    """
    if not paths:
        raise ValueError("At least one path must be provided.")

    if isinstance(paths, str):
        paths = [paths]

    # Build command string instead of list
    vulture_bin_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv/bin/vulture"
    command = f"{vulture_bin_path} {' '.join(paths)}"

    if exclude:
        command += f" --exclude {','.join(exclude)}"
    if ignore_decorators:
        command += f" --ignore-decorators {','.join(ignore_decorators)}"
    if ignore_names:
        command += f" --ignore-names {','.join(ignore_names)}"
    if make_whitelist:
        command += " --make-whitelist"
    if min_confidence is not None:
        command += f" --min-confidence {min_confidence}"
    if sort_by_size:
        command += " --sort-by-size"
    if config:
        command += f" --config {config}"
    if verbose:
        command += " --verbose"

    # Use run_command instead of subprocess
    errors: list[CodeValidationError] = []
    work_dir = os.path.dirname(__file__)

    # try:
    for line in run_command(command, work_dir=work_dir):
        if line.startswith("data: "):
            # Remove "data: " prefix
            content = line[len("data: "):].strip()
            # Add actual code string for unused code
            splitted_content = content.split(":")
            if len(splitted_content) == 3:
                file_path, line_num, error_msg = splitted_content
                if not os.path.exists(file_path):
                    continue

                with open(file_path, 'r') as f:
                    code = f.readlines()[int(line_num)-1].strip()

                errors.append({
                    "file": os.path.realpath(file_path),
                    "line": int(line_num),
                    "message": error_msg,
                    "code": code,
                })

    return errors or None


def main():
    """
    Main function to demonstrate usage of the check_unused_code function.
    """
    # Example 1: Analyze a single file
    logger.info("\nExample 1: Analyzing a single file")
    errors = check_unused_code(
        paths="mocks/sample_unused_code.py",
        min_confidence=20
    )
    if errors:
        for idx, error in enumerate(errors):
            logger.newline()
            logger.error(f"Unused code #{idx + 1}:\n{error['file']}\nLine: {
                         error['line']}\nError: {error['message']}\nCode: {error['code']}")
    else:
        logger.success("No unused code")
    # Example 2: Analyze a directory with exclusions and verbose mode
    logger.info("\nExample 2: Analyzing a directory")
    errors = check_unused_code(
        paths=["mocks/"],
        exclude=["venv", "tests"],
        verbose=True
    )
    if errors:
        for idx, error in enumerate(errors):
            logger.newline()
            logger.error(f"Unused code #{idx + 1}:\n{error['file']}\nLine: {
                         error['line']}\nError: {error['message']}\nCode: {error['code']}")
    else:
        logger.success("No unused code")


if __name__ == "__main__":
    main()
