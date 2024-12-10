import sys
import os
import fnmatch
import subprocess
from find_files import find_files
from utils import logger


# def match_pattern(file_path: str, pattern: str) -> bool:
#     if os.sep in pattern:
#         normalized_path = os.path.normpath(file_path)
#         normalized_pattern = os.path.normpath(pattern.lstrip('/'))
#         return fnmatch.fnmatch(normalized_path, f"*{normalized_pattern}")
#     else:
#         return fnmatch.fnmatch(os.path.basename(file_path), pattern)


# def has_content(file_path: str) -> bool:
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             return bool(file.read().strip())
#     except (IOError, UnicodeDecodeError):
#         return False


def deactivate_current_environment() -> None:
    if "VIRTUAL_ENV" in os.environ:
        deactivate_script = os.path.join(
            os.environ["VIRTUAL_ENV"], "bin", "deactivate")
        if os.path.exists(deactivate_script):
            subprocess.run(f"source {deactivate_script}",
                           shell=True, executable="/bin/bash")
        del os.environ["VIRTUAL_ENV"]


def reduce_path(base_path, n):
    # Split the path into components
    path_parts = base_path.rstrip(os.sep).split(os.sep)

    # Reduce the path by n levels
    reduced_path = os.sep.join(
        path_parts[:-n]) if n < len(path_parts) else base_path
    return reduced_path


def activate_nearest(current_dir: str, current_python_path: str) -> None:
    includes = ["*/bin/activate"]
    excludes = [
        "site-packages",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
    ]

    matching_files = find_files(
        current_dir, includes, excludes, limit=1)

    has_activated = bool(current_python_path)

    if matching_files:
        nearest_activation = matching_files[0]
        new_python_path = reduce_path(nearest_activation, n=3)

        if new_python_path in current_dir:
            if current_python_path != new_python_path:
                command = f"source {nearest_activation}"
                print(command)
                logger.success(
                    f"Activated virtual environment in: {nearest_activation}")
        elif has_activated:
            # logger.error("No virtual environment found.")
            print("deactivate")
    elif has_activated:
        # logger.error("No virtual environment found.")
        print("deactivate")


if __name__ == "__main__":
    current_dir = sys.argv[1]
    current_python_path = sys.argv[2]

    current_python_path = reduce_path(current_python_path, n=1)

    activate_nearest(current_dir, current_python_path)
