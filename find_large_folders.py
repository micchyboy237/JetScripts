from typing import Generator, List, Optional
import os
import shutil
import json
import argparse
from datetime import datetime
import fnmatch
from jet.file import traverse_directory
from jet.logger import logger
from jet.transformers.object import make_serializable


def match_patterns(file_path: str, patterns: List[str]) -> bool:
    """
    Matches a file path against a list of patterns.
    """
    normalized_path = os.path.normpath(file_path)
    return any(fnmatch.fnmatch(normalized_path, f"*{os.path.normpath(p)}") for p in patterns)


def get_size(file_path):
    return os.path.getsize(file_path)


def get_folder_sizes(folder_path):
    total_size = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += get_size(file_path)
            except FileNotFoundError:
                continue
    return total_size / (1000 * 1000)  # Convert to MB


def find_large_folders(base_dir, includes, excludes, min_size_mb, delete_folders=False, depth=3, **kwargs):
    """
    Find folders matching the criteria and optionally delete them.
    Accepts additional parameters through **kwargs and passes them to traverse_directory.
    """
    matched_folders = []

    depth = kwargs.pop(
        "max_forward_depth") if "max_forward_depth" in kwargs else depth

    # Pass **kwargs here
    for folder, current_depth in traverse_directory(base_dir, includes, excludes, max_forward_depth=depth, **kwargs):
        folder_size = get_folder_sizes(folder)
        if folder_size >= min_size_mb:
            logger.success(f"Folder ({current_depth}): {
                           folder} | Size: {folder_size:.2f} MB")
            matched_folders.append({"file": folder, "size": folder_size})
            if delete_folders:
                logger.warning(f"Deleting folder: {folder}")
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
    file_path, *arg_list = sys.argv
    transformed_args = []
    for arg in arg_list:
        arg_val = arg
        try:
            arg_val = int(arg)
        except (TypeError, ValueError) as e:
            logger.error(e)
            if not (isinstance(arg, str) and arg.startswith("-")):
                arg_val = f'"{arg}"'
        transformed_args.append(str(arg_val))
    command_args = " ".join([__file__] + transformed_args)
    return "python " + command_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and optionally delete large folders.")
    parser.add_argument(
        "-b", "--base-dir", type=str,
        help="Base directory to search for large folders. Defaults to current working directory.",
        default=os.getcwd()
    )
    parser.add_argument("-s", "--min-size", type=int, default=100,
                        help="Minimum size (MB) to consider a folder large.")
    parser.add_argument("-d", "--delete", action="store_true",
                        help="Enable deletion of matched folders.",
                        default=False)
    parser.add_argument("-i", "--includes", type=str,
                        help="Comma-separated list of patterns to include (e.g., '**/*,**/.git'). Defaults to all files.",
                        default="**/*")
    parser.add_argument("-e", "--excludes", type=str,
                        help="Comma-separated list of patterns to exclude (e.g., 'node_modules,.venv,*.env').",
                        default="")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Maximum number of folder paths to yield.")
    parser.add_argument("-f", "--output-file", type=str, default=None,
                        help="Optional path to save results as a JSON file.")
    parser.add_argument("--direction", type=str,
                        choices=["forward", "backward", "both"], default="forward",
                        help="Direction of traversal - 'forward' (default), 'backward', or 'both'.")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Maximum depth to traverse forward.")
    parser.add_argument("--max-backward-depth", type=int, default=None,
                        help="Maximum depth to traverse upwards (for 'backward' or 'both').")

    args = parser.parse_args()
    command = get_command()
    logger.log("COMMAND:", command or "[]", colors=["WHITE", "INFO"])

    includes = [item for item in args.includes.split(",") if item]
    excludes = [item for item in args.excludes.split(",") if item]

    # Pass new arguments to the find_large_folders function using **kwargs
    results = find_large_folders(
        args.base_dir, includes, excludes, args.min_size, args.delete,
        limit=args.limit, direction=args.direction, depth=args.max_depth, max_backward_depth=args.max_backward_depth)

    total_size = calculate_total_size(results)
    total_size = format_size(total_size)

    formatted_data = {
        "date": datetime.now().strftime("%b %d, %Y | %I:%M %p"),
        "command": command,
        "base_dir": args.base_dir,
        "includes": includes,
        "excludes": excludes,
        "count": len(results),
        "total_size": total_size,
        "deleted": args.delete,
        "limit": args.limit,
        "min_size": args.min_size,
        "max_depth": args.max_depth,
        "results": sorted(results, key=lambda x: x["size"], reverse=True),
    }

    if args.output_file:
        save_to_json(formatted_data, args.output_file)

    if args.delete:
        print(f"Total Freed Space: {total_size}")
    else:
        print(f"Total Size: {total_size}")

# Commands
# python find_large_folders.py -s 100 -i "aim,.aim"
# python find_large_folders.py -s 0 -i "<folder>/bin/activate"
# python find_large_folders.py -s 0 -i "<folder>/bin/activate" -b "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts"
# python find_large_folders.py -s 100 -b "/path/to/base/dir" -i "<folder>/node_modules" -e "node_modules/**"
