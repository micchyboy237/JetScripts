from typing import Generator, List, Optional
import os
import shutil
import argparse
import fnmatch
from tqdm import tqdm
from jet.file import traverse_directory
from jet.file.utils import save_file
from jet.logger import logger


def match_patterns(file_path: str, patterns: List[str]) -> bool:
    """Check if a file path matches any of the given patterns (case-insensitive)."""
    normalized_path = os.path.normpath(file_path).lower()
    return any(fnmatch.fnmatch(normalized_path, f"*{os.path.normpath(p).lower()}") for p in patterns)


def get_size(file_path: str) -> int:
    """Get the size of a file in bytes."""
    return os.path.getsize(file_path)


def get_folder_sizes(folder_path: str) -> float:
    """Calculate the total size of a folder in MB."""
    total_size = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += get_size(file_path)
            except FileNotFoundError:
                continue
    return total_size / (1024 * 1024)  # Use 1024-based MB for consistency


def find_large_folders(
    base_dir: str,
    includes: List[str],
    excludes: List[str],
    min_size_mb: int,
    delete_folders: bool = False,
    depth: Optional[int] = None,
    **kwargs,
) -> Generator[dict, None, List[dict]]:
    """Find folders larger than min_size_mb and optionally delete them.

    Yields individual folder data and finally returns the complete sorted list.
    """
    results: List[dict] = []
    base_dir = os.path.expanduser(base_dir)
    output_file: str = kwargs.pop("output_file", os.path.join(base_dir, "_large_folders.json"))
    save_results: bool = kwargs.pop("save", False)

    total_folders = 0
    pbar = tqdm(desc="Scanning folders", unit=" folder")

    # Configure traversal depth/direction
    direction: str = kwargs.get("direction", "forward")
    kwargs["max_forward_depth"] = depth if direction in ("forward", "both") else None
    kwargs["max_backward_depth"] = kwargs.get("max_backward_depth") if direction in ("backward", "both") else None

    for folder, current_depth in traverse_directory(base_dir, includes, excludes, **kwargs):
        folder_size_mb = get_folder_sizes(folder)
        if folder_size_mb >= min_size_mb:
            total_folders += 1
            pbar.set_postfix({"Depth": current_depth, "Large folders": total_folders})
            pbar.update(1)

            logger.success(f"\nSize: {format_size(folder_size_mb)} | Folder: {folder}")

            folder_data = {"size": folder_size_mb, "file": folder, "depth": current_depth}
            results.append(folder_data)
            results.sort(key=lambda x: x["size"], reverse=True)

            if save_results:
                save_intermediate_results(results, output_file, min_size_mb, depth, kwargs.get("max_backward_depth"))

            yield folder_data

            if delete_folders:
                logger.warning(f"Deleting folder: {folder}")
                shutil.rmtree(folder)

    pbar.close()
    return results


def save_intermediate_results(
    results: List[dict],
    output_file: str,
    min_size_mb: int,
    depth: Optional[int],
    max_backward_depth: Optional[int],
) -> None:
    """Save current results to JSON (used for live updates during long scans)."""
    final_results = {
        "file": output_file,
        "size": calculate_total_size(results),
        "min_size_mb": min_size_mb,
        "depth": depth,
        "max_backward_depth": max_backward_depth,
        "count": len(results),
        "results": results,
    }
    save_file(final_results, output_file)
    logger.info(f"Updated output file: {output_file}")


def format_size(size_mb: float) -> str:
    """Format size in MB to a human-readable string (MB or GB)."""
    if size_mb >= 1000:
        return f"{size_mb / 1000:.2f} GB"
    return f"{size_mb:.2f} MB"


def calculate_total_size(items: List[dict]) -> float:
    """Calculate the total size in MB of all found large folders."""
    return sum(item["size"] for item in items)


def get_command() -> str:
    """Get the command-line string used to run the script."""
    import sys

    file_path, *arg_list = sys.argv
    transformed_args = []
    for arg in arg_list:
        try:
            int(arg)  # Numeric args stay as-is
            transformed_args.append(arg)
        except ValueError:
            if arg.startswith("-"):
                transformed_args.append(arg)
            else:
                transformed_args.append(f'"{arg}"')
    command_args = " ".join([sys.argv[0]] + transformed_args)
    return "python " + command_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and optionally delete large folders.")
    parser.add_argument("-b", "--base-dir", type=str, default=os.getcwd(),
                        help="Base directory to search. Defaults to current working directory.")
    parser.add_argument("-s", "--min-size", type=int, default=50,
                        help="Minimum size (MB) to consider a folder large.")
    parser.add_argument("-d", "--max-depth", type=int, default=None,
                        help="Maximum forward depth. Set to 0 for immediate subdirectories only.")
    parser.add_argument("-i", "--includes", type=str,
                        default="*cache*,*Cache*,*CACHE*,*tmp*,*Temp*,.TemporaryItems,Temporary Files,.Spotlight-V100,.fseventsd,.DS_Store,Logs,DerivedData,generated,node_modules,__pycache__,dist,build,.venv,*_venv,.pytest_cache",
                        help="Comma-separated include patterns.")
    parser.add_argument("-e", "--excludes", type=str, default="",
                        help="Comma-separated exclude patterns.")
    parser.add_argument("-f", "--output-file", type=str, default=None,
                        help="Optional JSON output file path.")
    parser.add_argument("--max-backward-depth", type=int, default=None,
                        help="Maximum upward depth when direction is backward or both.")
    parser.add_argument("--delete", action="store_true",
                        help="Delete matched folders (dangerous – use with caution).")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON file (updates live during scan).")
    parser.add_argument("--direction", type=str, choices=["forward", "backward", "both"],
                        default="forward", help="Traversal direction.")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Limit number of yielded results (not implemented yet).")

    args = parser.parse_args()

    command = get_command()
    logger.log("COMMAND:", command or "[]", colors=["WHITE", "INFO"])

    includes = [p.strip() for p in args.includes.split(",") if p.strip()]
    excludes = [p.strip() for p in args.excludes.split(",") if p.strip()]

    output_file = args.output_file or os.path.join(args.base_dir, "_large_folders.json")

    results = []
    generator = find_large_folders(
        base_dir=args.base_dir,
        includes=includes,
        excludes=excludes,
        min_size_mb=args.min_size,
        delete_folders=args.delete,
        depth=args.max_depth,
        direction=args.direction,
        max_backward_depth=args.max_backward_depth,
        output_file=output_file,
        save=args.save,
    )

    logger.info(f"Output file: {output_file}")

    for folder_data in generator:
        results.append(folder_data)
        results.sort(key=lambda x: x["size"], reverse=True)

    total_size_mb = calculate_total_size(results)
    formatted_total = format_size(total_size_mb)

    if args.save:
        final_results = {
            "file": output_file,
            "size": total_size_mb,
            "min_size_mb": args.min_size,
            "depth": args.max_depth,
            "max_backward_depth": args.max_backward_depth,
            "count": len(results),
            "results": results,
        }
        save_file(final_results, output_file)

    if args.delete:
        print(f"Total Freed Space: {formatted_total}")
    else:
        print(f"Total Size of top-level large folders: {formatted_total}")