import os
import fnmatch
from typing import List
from tqdm import tqdm
from jet.logger import logger

INCLUDES = [
    ".env.enter",
    "*/bin/activate"  # Matches files named "activate" under any "bin" folder
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
    "build",
    "dist",
    "logs",
    "tmp",
    "temp",
    "coverage",
]


def match_pattern(file_path: str, pattern: str) -> bool:
    """
    Matches a file path against a pattern that can include folder components.

    :param file_path: The full path of the file to match.
    :param pattern: The pattern to match against.
    :return: True if the pattern matches the file path, otherwise False.
    """
    if os.sep in pattern:  # Pattern includes folder components
        # Normalize paths for consistent matching
        normalized_path = os.path.normpath(file_path)
        normalized_pattern = os.path.normpath(pattern.lstrip('/'))
        return fnmatch.fnmatch(normalized_path, f"*{normalized_pattern}")
    else:
        # Standard file matching
        return fnmatch.fnmatch(os.path.basename(file_path), pattern)


def has_content(file_path: str) -> bool:
    """
    Checks if the file has any non-whitespace content.

    :param file_path: The path to the file to check.
    :return: True if the file has non-whitespace content, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the file content and strip whitespace
            return bool(file.read().strip())
    except (IOError, UnicodeDecodeError):
        # In case of an error opening or reading the file, assume no content
        return False


def find_files(starting_dir: str, includes: List[str], excludes: List[str] = [], limit: int = None) -> List[str]:
    """
    Finds files in a directory and its subdirectories that match include patterns,
    while excluding those that match exclude patterns. Stops when the matched file count hits the limit.

    :param starting_dir: The starting directory to search in.
    :param includes: A list of wildcard patterns to include (e.g., ["*.py", "*.txt"]).
                     Must not be empty.
    :param excludes: A list of wildcard patterns to exclude (e.g., ["test_*.py"]).
    :param limit: An optional limit on the number of files to match.
    :return: A list of matching file paths.
    :raises ValueError: If the includes list is empty.
    """
    if not includes:
        raise ValueError("The includes list must not be empty.")

    matching_files = []
    matched_count = 0  # Track the number of matched files

    # Walk through the directory tree with tqdm progress bar
    with tqdm(total=0, desc="Processing files", unit=" file", dynamic_ncols=True) as pbar:
        for root, dirs, files in os.walk(starting_dir):
            # Exclude directories that start with a dot
            # dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                file_path = os.path.join(root, file)

                # Check if the file matches any of the include patterns
                if any(match_pattern(file_path, pattern) for pattern in includes):
                    # Check if the file matches any of the exclude patterns
                    if not any(match_pattern(file_path, pattern) for pattern in excludes):
                        # Check if the file has content
                        if has_content(file_path):
                            logger.debug(f"\nMatched: {file_path}")
                            matching_files.append(file_path)
                            matched_count += 1  # Increment the matched file count

                            # Update progress only with the matched files count, not every file
                            pbar.set_postfix(
                                matched=matched_count, refresh=True)
                            pbar.update(1)

                            # If a limit is set, stop once the limit is reached
                            if limit and matched_count >= limit:
                                pbar.set_postfix(
                                    matched=matched_count, limit_reached=True, refresh=True)
                                return matching_files

    return matching_files


# Example usage:
if __name__ == "__main__":
    starting_directory = "/Users/jethroestrada/Desktop/External_Projects"
    include_patterns = INCLUDES
    exclude_patterns = EXCLUDES
    limit = None  # Optional limit

    try:
        matching_files = find_files(starting_directory,
                                    include_patterns, exclude_patterns, limit)
        logger.log("Matching files:")
        for file in matching_files:
            logger.success(file)
    except ValueError as e:
        logger.error(e)
