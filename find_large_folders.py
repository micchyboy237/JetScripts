from typing import Generator, List, Optional
import os
import shutil
import json
import argparse
from datetime import datetime
import fnmatch
from tqdm import tqdm
from jet.file import traverse_directory
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.object import make_serializable


def match_patterns(file_path: str, patterns: List[str]) -> bool:
    """Check if a file path matches any of the given patterns."""
    normalized_path = os.path.normpath(file_path)
    return any(fnmatch.fnmatch(normalized_path, f"*{os.path.normpath(p)}") for p in patterns)


def get_size(file_path):
    """Get the size of a file in bytes."""
    return os.path.getsize(file_path)


def get_folder_sizes(folder_path):
    """Calculate the total size of a folder in MB."""
    total_size = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += get_size(file_path)
            except FileNotFoundError:
                continue
    return total_size / (1000 * 1000)


def find_large_folders(base_dir, includes, excludes, min_size_mb, delete_folders=False, depth: Optional[int] = 2, **kwargs) -> Generator[dict, None, List[dict]]:
    """Find folders larger than min_size_mb and optionally delete them."""
    results = []
    output_file = kwargs.pop("output_file", os.path.join(
        base_dir, "_large_folders.json"))
    save_results = kwargs.pop("save", False)
    total_folders = 0
    accumulated_size = 0
    pbar = tqdm(desc="Scanning folders", unit=" folder")
    # Pass depth via kwargs to avoid duplicate argument
    kwargs["max_forward_depth"] = depth
    for folder, current_depth in traverse_directory(base_dir, includes, excludes, **kwargs):
        folder_size = get_folder_sizes(folder)
        if folder_size >= min_size_mb:
            total_folders += 1
            if current_depth == 0:
                accumulated_size += folder_size
            pbar.set_postfix(
                {"Depth": current_depth, "Folders": total_folders})
            pbar.update(1)
            logger.success(
                f"\nSize: {format_size(folder_size)} | Folder: {folder}")
            folder_data = {"size": folder_size,
                           "file": folder, "depth": current_depth}
            results.append(folder_data)
            results.sort(key=lambda x: x["size"], reverse=True)
            if save_results:
                final_results = {
                    "file": output_file,
                    "size": calculate_total_size(results),
                    "min_size_mb": min_size_mb,
                    "depth": depth,
                    "max_backward_depth": kwargs.get("max_backward_depth"),
                    "results": results
                }
                save_file(final_results, output_file)
                logger.info(f"Updated output file: {output_file}")
            yield folder_data
            if delete_folders:
                logger.warning(f"Deleting folder: {folder}")
                shutil.rmtree(folder)
    pbar.close()
    return results


def format_size(size_mb):
    """Format size in MB to a human-readable string (MB or GB)."""
    if size_mb >= 1000:
        return f"{size_mb / 1000:.2f} GB"
    return f"{size_mb:.2f} MB"


def calculate_total_size(deleted_folders: list[dict]) -> float:
    """Calculate the total size of top-level folders only."""
    total_size = 0
    for item in deleted_folders:
        if item["depth"] == 0:
            total_size += item["size"]
    return total_size


def get_command() -> str:
    """Get the command-line string used to run the script."""
    import sys
    file_path, *arg_list = sys.argv
    transformed_args = []
    for arg in arg_list:
        arg_val = arg
        try:
            arg_val = int(arg)
        except (TypeError, ValueError) as e:
            if not (isinstance(arg, str) and arg.startswith("-")):
                arg_val = f'"{arg}"'
        transformed_args.append(str(arg_val))
    command_args = " ".join([__file__] + transformed_args)
    return "python " + command_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and optionally delete large folders.")
    parser.add_argument("-b", "--base-dir", type=str,
                        help="Base directory to search for large folders. Defaults to current working directory.", default=os.getcwd())
    parser.add_argument("-s", "--min-size", type=int, default=100,
                        help="Minimum size (MB) to consider a folder large.")
    parser.add_argument("-d", "--max-depth", type=int, default=1,
                        help="Maximum depth to traverse forward. Defaults to 1. Set to 0 to list only immediate subdirectories.")
    parser.add_argument("-i", "--includes", type=str, help="Comma-separated list of patterns to include.",
                        default="*cache*,*Cache*,*CACHE*,*tmp*,*Temp*,.TemporaryItems,Temporary Files,.Spotlight-V100,.fseventsd,.DS_Store,Logs,DerivedData")
    parser.add_argument("-e", "--excludes", type=str,
                        help="Comma-separated list of patterns to exclude.", default="")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Maximum number of folder paths to yield.")
    parser.add_argument("-f", "--output-file", type=str, default=None,
                        help="Optional path to save results as a JSON file.")
    parser.add_argument("--max-backward-depth", type=int, default=None,
                        help="Maximum depth to traverse upwards (for 'backward' or 'both').")
    parser.add_argument("--delete", action="store_true",
                        help="Enable deletion of matched folders.", default=False)
    parser.add_argument("--save", action="store_true",
                        help="Enable saving the results to the output file.", default=False)
    parser.add_argument("--direction", type=str, choices=["forward", "backward", "both"],
                        default="forward", help="Direction of traversal - 'forward' (default), 'backward', or 'both'.")
    args = parser.parse_args()
    command = get_command()
    logger.log("COMMAND:", command or "[]", colors=["WHITE", "INFO"])
    includes = [item for item in args.includes.split(",") if item]
    excludes = [item for item in args.excludes.split(",") if item]
    output_file = args.output_file or os.path.join(
        args.base_dir, "_large_folders.json")
    results = []
    generator = find_large_folders(
        args.base_dir, includes, excludes, args.min_size, args.delete, depth=args.max_depth,
        direction=args.direction, max_backward_depth=args.max_backward_depth,
        output_file=output_file, save=args.save
    )
    logger.info(f"Output file: {output_file}")
    for folder_data in generator:
        results.append(folder_data)
        results.sort(key=lambda x: x["size"], reverse=True)
    total_size = calculate_total_size(results)
    final_results = {
        "file": output_file,
        "size": total_size,
        "min_size_mb": args.min_size,
        "depth": args.max_depth,
        "max_backward_depth": args.max_backward_depth,
        "results": results
    }
    if args.save:
        save_file(final_results, output_file)
    total_size = format_size(total_size)
    if args.delete:
        print(f"Total Freed Space: {total_size}")
    else:
        print(f"Total Size: {total_size}")
