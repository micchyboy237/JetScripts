from typing import Generator, List, Optional
import os
import shutil
import json
import argparse
from datetime import datetime
import fnmatch
from jet.logger import logger


def traverse_directory(
    base_dir: str,
    includes: List[str],
    excludes: List[str] = [],
    limit: Optional[int] = None,
    direction: str = "forward",
    max_backward_depth: Optional[int] = None
) -> Generator[str, None, None]:
    """
    Generator that traverses directories and yields folder paths 
    matching the include patterns but not the exclude patterns.

    :param base_dir: The directory to start traversal from.
    :param includes: Patterns to include in the search.
    :param excludes: Patterns to exclude from the search.
    :param limit: Maximum number of folder paths to yield.
    :param direction: Direction of traversal - 'forward' (default), 'backward', or 'both'.
    :param max_backward_depth: Maximum depth to traverse upwards (for 'backward' or 'both').
    """
    visited_paths = set()  # Prevent circular references
    yielded_count = 0
    current_depth = 0
    current_dir = os.path.abspath(base_dir)

    def match_patterns(path: str, patterns: List[str]) -> bool:
        """Checks if a path matches any of the given patterns."""
        for pattern in patterns:
            if "<folder>" in pattern:
                folder_path = os.path.join(
                    path, pattern.replace("<folder>", "").lstrip("/"))
                if os.path.exists(folder_path):
                    return True
            elif fnmatch.fnmatch(path, f"*{os.path.normpath(pattern.lstrip('/'))}"):
                return True
        return False

    def search_dir(directory: str) -> Generator[str, None, None]:
        """Traverses a single directory and yields matching paths."""
        nonlocal yielded_count
        for root, dirs, _ in os.walk(directory, followlinks=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                real_path = os.path.realpath(folder_path)

                if real_path in visited_paths:
                    continue
                visited_paths.add(real_path)

                if match_patterns(folder_path, excludes) or any(exclude in folder_path for exclude in excludes):
                    continue
                if match_patterns(folder_path, includes) or any(include in folder_path for include in includes):
                    yield folder_path
                    yielded_count += 1
                    if limit and yielded_count >= limit:
                        return

    # Traverse forward
    if direction in {"forward", "both"}:
        yield from search_dir(current_dir)
        if limit and yielded_count >= limit:
            return

    # Traverse backward
    if direction in {"backward", "both"}:
        while True:
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Root directory reached
                break
            current_dir = parent_dir
            current_depth += 1
            if max_backward_depth is not None and current_depth > max_backward_depth:
                break
            yield from search_dir(current_dir)
            if limit and yielded_count >= limit:
                return


def match_patterns(file_path: str, patterns: List[str]) -> bool:
    """
    Matches a file path against a list of patterns.
    """
    normalized_path = os.path.normpath(file_path)
    return any(fnmatch.fnmatch(normalized_path, f"*{os.path.normpath(p)}") for p in patterns)


def get_folder_size(folder_path):
    total_size = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except FileNotFoundError:
                continue
    return total_size / (1000 * 1000)  # Convert to MB


def find_large_folders(base_dir, includes, excludes, min_size_mb, delete_folders=False, **kwargs):
    """
    Find folders matching the criteria and optionally delete them.
    Accepts additional parameters through **kwargs and passes them to traverse_directory.
    """
    matched_folders = []

    # Pass **kwargs here
    for folder in traverse_directory(base_dir, includes, excludes, **kwargs):
        folder_size = get_folder_size(folder)
        if folder_size >= min_size_mb:
            print(f"Folder: {folder} | Size: {folder_size:.2f} MB")
            matched_folders.append({"file": folder, "size": folder_size})
            if delete_folders:
                print(f"Deleting folder: {folder}")
                shutil.rmtree(folder)

    return matched_folders


def save_to_json(data, output_file: str):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to '{output_file}'")


def format_size(size_mb):
    if size_mb >= 1000:
        return f"{size_mb / 1000:.2f} GB"
    return f"{size_mb:.2f} MB"


def calculate_total_size(deleted_folders: list[dict]) -> float:
    total_size = 0
    for item in deleted_folders:
        total_size += item["size"]
    return total_size


def get_command() -> str:
    import sys
    return " ".join(sys.argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and optionally delete large folders.")
    parser.add_argument("-b", "--base-dir", type=str,
                        help="Base directory to search for large folders. Defaults to '/Users/jethroestrada/Desktop/External_Projects'.",
                        default="/Users/jethroestrada/Desktop/External_Projects")
    parser.add_argument("-s", "--min-size", type=int, default=100,
                        help="Minimum size (MB) to consider a folder large.")
    parser.add_argument("-d", "--delete", action="store_true",
                        help="Enable deletion of matched folders.")
    parser.add_argument("-i", "--includes", type=str,
                        help="Comma-separated list of patterns to include (e.g., '**/*,**/.git'). Defaults to all files.",
                        default="**/*")
    parser.add_argument("-e", "--excludes", type=str,
                        help="Comma-separated list of patterns to exclude (e.g., 'node_modules,.venv,*.env').",
                        default="")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of folder paths to yield.")
    parser.add_argument("--direction", type=str,
                        choices=["forward", "backward", "both"], default="forward",
                        help="Direction of traversal - 'forward' (default), 'backward', or 'both'.")
    parser.add_argument("--max-backward-depth", type=int, default=None,
                        help="Maximum depth to traverse upwards (for 'backward' or 'both').")

    args = parser.parse_args()

    includes = [item for item in args.includes.split(",") if item]
    excludes = [item for item in args.excludes.split(",") if item]

    # Pass new arguments to the find_large_folders function using **kwargs
    results = find_large_folders(
        args.base_dir, includes, excludes, args.min_size, args.delete,
        limit=args.limit, direction=args.direction, max_backward_depth=args.max_backward_depth)

    total_size = calculate_total_size(results)
    total_size = format_size(total_size)

    formatted_data = {
        "date": datetime.now().strftime("%b %d, %Y | %I:%M %p"),
        "command": get_command(),
        "base_dir": args.base_dir,
        "includes": includes,
        "excludes": excludes,
        "count": len(results),
        "total_size": total_size,
        "deleted": args.delete,
        "limit": args.limit,
        "min_size": args.min_size,
        "results": sorted(results, key=lambda x: x["size"], reverse=True),
    }

    output_dir = "generated/find_large_folders"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = "deleted-folders" if args.delete else "searched-folders"
    output_file = f"{output_dir}/{file_name}.json"

    save_to_json(formatted_data, output_file)
    if args.delete:
        print(f"Total Freed Space: {total_size}")
    else:
        print(f"Total Size: {total_size}")

# Commands
# python find_large_folders.py -s 0 -i "<folder>/bin/activate"
# python find_large_folders.py -s 100 -b "/path/to/base/dir" -i "<folder>/node_modules" -e "node_modules/**"
