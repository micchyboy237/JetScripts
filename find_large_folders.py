import os
import shutil
import json
import argparse
from datetime import datetime
import fnmatch

INCLUDES = [
    # "**/.git",
    # "node_modules",
    "**/*"
]

EXCLUDES = [
    # ".venv",
]


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


def format_size(size_mb):
    if size_mb >= 1000:
        return f"{size_mb / 1000:.2f} GB"
    return f"{size_mb:.2f} MB"


def match_patterns(folder_path, patterns):
    """Match folder_path against a list of patterns."""
    for pattern in patterns:
        if "**" in pattern:
            # Simulate '**' behavior
            if fnmatch.fnmatch(folder_path, pattern.replace("**", "*")):
                return True
        else:
            # Match absolute paths or base names using fnmatch
            if fnmatch.fnmatch(folder_path, pattern) or fnmatch.fnmatch(os.path.basename(folder_path), pattern):
                return True
    return False


def find_large_folders(base_dir, min_size_mb=200, delete_folders=False):
    min_size = min_size_mb
    matched_folders = []
    visited_paths = set()  # Keep track of visited paths

    def traverse_directory(current_dir):
        if os.path.realpath(current_dir) in visited_paths:  # Avoid circular reference
            return

        visited_paths.add(os.path.realpath(current_dir))

        for entry in os.scandir(current_dir):
            if not entry.is_dir(follow_symlinks=False):
                continue

            folder = entry.path
            if match_patterns(folder, EXCLUDES):  # Skip excluded folders
                continue

            if match_patterns(folder, INCLUDES):  # Check inclusion
                folder_size = get_folder_size(folder)

                if folder_size >= min_size:
                    print(
                        f"-----------\nFolder: {folder}\nSize: {format_size(folder_size)}")
                    matched_folders.append(
                        {"file": folder, "size": folder_size})
                    if delete_folders:
                        print(f"Deleting folder: {folder}")
                        shutil.rmtree(folder)
                else:
                    traverse_directory(folder)  # Recurse if size is smaller
            else:
                traverse_directory(folder)

    traverse_directory(base_dir)

    return matched_folders


def save_to_json(data, output_dir="generated"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"{output_dir}/deleted-folders-{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to '{output_path}'")


def calculate_total_size(deleted_folders: list[dict]) -> float:
    total_size = 0
    for item in deleted_folders:
        total_size += item["size"]
    return total_size


def parse_comma_separated_list(arg):
    """Parses a comma-separated list of patterns from the CLI."""
    return [item.strip() for item in arg.split(",") if item.strip()]


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Find and optionally delete large folders."
    )
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
                        default="site-packages,.venv,*.env")

    args = parser.parse_args()

    base_dir = args.base_dir
    min_size_mb = args.min_size
    delete_folders = args.delete

    # Parse include and exclude patterns
    INCLUDES = parse_comma_separated_list(args.includes)
    EXCLUDES = parse_comma_separated_list(args.excludes)

    large_folders = find_large_folders(base_dir, min_size_mb, delete_folders)
    print(f"Count: ({len(large_folders)})")

    total_size = calculate_total_size(large_folders)
    total_size = format_size(total_size)
    # Format the JSON with count, total freed space, and results
    formatted_data = {
        "base_dir": base_dir,
        "count": len(large_folders),
        "total_size": total_size,
        "deleted": delete_folders,
        "results": sorted(large_folders, key=lambda x: x["size"], reverse=True),
    }

    # Save the results to JSON
    save_to_json(formatted_data)
    if delete_folders:
        print(f"Total Freed Space: {total_size}")
    else:
        print(f"Total Size: {total_size}")

# Example command:
# python find_large_folders.py -b "/path/to/dir" -s 100 -d -i "**/*" -e "node_modules,.venv,*.env"
