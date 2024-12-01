import os
import shutil
import json
import argparse
import fnmatch
from typing import Generator, List, Optional
from datetime import datetime
from jet.time.utils import get_file_dates


def traverse_directory(
    base_dir: str,
    includes: List[str],
    excludes: List[str] = [],
    limit: Optional[int] = None,
    direction: str = "forward",
    max_backward_depth: Optional[int] = None
) -> Generator[str, None, None]:
    visited_paths = set()
    yielded_count = 0
    current_depth = 0
    current_dir = os.path.abspath(base_dir)

    def match_patterns(path: str, patterns: List[str]) -> bool:
        for pattern in patterns:
            wildcard_pattern = pattern.replace("<folder>", "*")
            if fnmatch.fnmatch(path, f"*{os.path.normpath(wildcard_pattern)}"):
                return True
        return False

    def search_dir(directory: str) -> Generator[str, None, None]:
        nonlocal yielded_count
        for root, dirs, _ in os.walk(directory, followlinks=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if folder_path in visited_paths:
                    continue
                visited_paths.add(folder_path)
                if match_patterns(folder_path, excludes):
                    continue
                if match_patterns(folder_path, includes):
                    yield folder_path
                    yielded_count += 1
                    if limit and yielded_count >= limit:
                        return

    if direction in {"forward", "both"}:
        yield from search_dir(current_dir)
        if limit and yielded_count >= limit:
            return

    if direction in {"backward", "both"}:
        while True:
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir
            current_depth += 1
            if max_backward_depth is not None and current_depth > max_backward_depth:
                break
            yield from search_dir(current_dir)
            if limit and yielded_count >= limit:
                return


def get_folder_metadata(folder_path: str) -> dict:
    metadata = {}
    metadata.update(get_file_dates(folder_path, "%Y-%m-%d %H:%M:%S"))
    return metadata


def get_folder_size(folder_path: str) -> float:
    total_size = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except FileNotFoundError:
                continue
    return total_size / (1000 * 1000)  # Convert to MB


def calculate_total_size(deleted_folders: list[dict]) -> float:
    total_size = 0
    for item in deleted_folders:
        total_size += item["size"]
    return total_size


def format_size(size_mb):
    if size_mb >= 1000:
        return f"{size_mb / 1000:.2f} GB"
    return f"{size_mb:.2f} MB"


def find_large_folders(
    base_dir, includes, excludes, min_size_mb, delete_folders=False, include_metadata=False, **kwargs
):
    """
    Find folders matching the criteria and optionally delete them.
    Accepts additional parameters through **kwargs and passes them to traverse_directory.
    """
    matched_folders = []

    for folder in traverse_directory(base_dir, includes, excludes, **kwargs):
        folder_size = get_folder_size(folder)
        if folder_size >= min_size_mb:
            folder_info = {"folder": folder, "size": folder_size}
            if include_metadata:
                folder_info.update(get_folder_metadata(folder))
            print(
                f"{"Delete" if delete_folders else "Folder"}: {folder} | {format_size(folder_size)}")
            matched_folders.append(folder_info)
            if delete_folders:
                shutil.rmtree(folder)

    return matched_folders


def save_to_json(data, output_file: str):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to '{output_file}'")


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
    parser.add_argument("--include-metadata", action="store_true",
                        help="Include folder metadata in results.",
                        default=True)

    args = parser.parse_args()

    includes = [item.strip() for item in args.includes.split(",") if item]
    excludes = [item.strip() for item in args.excludes.split(",") if item]

    results = find_large_folders(
        args.base_dir, includes, excludes, args.min_size, args.delete, include_metadata=args.include_metadata,
        limit=args.limit, direction=args.direction, max_backward_depth=args.max_backward_depth
    )

    total_size = calculate_total_size(results)
    total_size = format_size(total_size)
    formatted_data = {
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

    output_dir = "generated"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_prefix = "deleted-folders" if args.delete else "searched-folders"
    output_file = f"{output_dir}/{file_prefix}-{timestamp}.json"

    save_to_json(formatted_data, output_file)
    if args.delete:
        print(f"Total Freed Space: {total_size}")
    else:
        print(f"Total Size: {total_size}")


# Commands
# python find_large_folders.py -s 0 -i "<folder>/bin/activate"
