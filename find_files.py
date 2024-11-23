import os
import fnmatch
from typing import List
from utils import logger

INCLUDES = [
    "*/bin/activate"
]
EXCLUDES = [
    "site-packages",
    "node_modules",
    "dist",
    "build",
    "cache",
    "ios",
    "android",
    "public/static",
    "__pycache__",
    "logs",
    "tmp",
    "temp",
    "coverage",
]


def match_pattern(file_path: str, pattern: str) -> bool:
    """
    Matches a file path against a pattern that can include folder components.
    """
    normalized_path = os.path.normpath(file_path)
    normalized_pattern = os.path.normpath(pattern.lstrip('/'))
    return fnmatch.fnmatch(normalized_path, f"*{normalized_pattern}")


def has_content(file_path: str) -> bool:
    """
    Checks if the file has any non-whitespace content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return bool(file.read().strip())
    except (IOError, UnicodeDecodeError):
        return False


def find_files(
    starting_dir: str,
    includes: List[str],
    excludes: List[str] = [],
    limit: int = None,
    direction: str = "both",
    max_backward_depth: int = None
) -> List[str]:
    """
    Finds files in a directory based on include and exclude patterns.

    :param starting_dir: The directory to start searching from.
    :param includes: Patterns to include in the search.
    :param excludes: Patterns to exclude from the search.
    :param limit: Maximum number of files to return.
    :param direction: Search direction - 'forward', 'backward', or 'both' (default).
    :param max_backward_depth: Maximum depth to search upwards (for 'backward' or 'both').
    :return: A list of matching file paths.
    """
    if not includes:
        raise ValueError("The includes list must not be empty.")
    if direction not in {"forward", "backward", "both"}:
        raise ValueError(
            "The direction must be one of 'forward', 'backward', or 'both'.")

    matching_files = []
    matched_count = 0
    current_depth = 0
    current_dir = os.path.abspath(starting_dir)

    def search_dir(directory):
        nonlocal matched_count
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if any(match_pattern(file_path, pattern) for pattern in includes):
                    if not any(match_pattern(file_path, pattern) for pattern in excludes):
                        if has_content(file_path):
                            # logger.debug(f"Matched: {file_path}")
                            matching_files.append(file_path)
                            matched_count += 1
                            if limit and matched_count >= limit:
                                # logger.warning(f"Limit of {limit} reached")
                                return True
        return False

    # Search forward
    if direction in {"forward", "both"}:
        if search_dir(current_dir):
            return matching_files

    # Search backward
    if direction in {"backward", "both"}:
        while True:
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir
            current_depth += 1
            if max_backward_depth is not None and current_depth > max_backward_depth:
                break
            if search_dir(current_dir):
                return matching_files

    logger.debug(f"Total matched files: {matched_count}")
    return matching_files


if __name__ == "__main__":
    # starting_directory = "/Users/jethroestrada/Desktop/External_Projects/AI/eval_agents/helicone"
    starting_directory = "/Users/jethroestrada/Desktop/External_Projects/AI/eval_agents/helicone/examples/postHog"
    include_patterns = INCLUDES
    exclude_patterns = EXCLUDES
    limit = 1
    direction = "both"
    max_backward_depth = 3
    try:
        matching_files = find_files(
            starting_directory,
            include_patterns,
            exclude_patterns,
            limit,
            direction=direction,
            max_backward_depth=max_backward_depth
        )
        logger.debug(f"Matching files: {len(matching_files)}")
    except ValueError as e:
        logger.error(e)
