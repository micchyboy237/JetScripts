import sys
import os
from find_files import find_files
from jet.file import traverse_directory
# from utils import logger
from jet.logger import logger


# def deactivate_current_environment() -> None:
#     if "VIRTUAL_ENV" in os.environ:
#         deactivate_script = os.path.join(
#             os.environ["VIRTUAL_ENV"], "bin", "deactivate")
#         if os.path.exists(deactivate_script):
#             subprocess.run(f"source {deactivate_script}",
#                            shell=True, executable="/bin/bash")
#         del os.environ["VIRTUAL_ENV"]


def reduce_path(base_path, n):
    # Split the path into components
    path_parts = base_path.rstrip(os.sep).split(os.sep)

    # Reduce the path by n levels
    reduced_path = os.sep.join(
        path_parts[:-n]) if n < len(path_parts) else base_path
    return reduced_path


def activate_nearest(current_dir: str, current_python_path: str, virtual_python_path: str) -> None:
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
                return nearest_activation
        elif has_activated:
            raise Exception("No virtual environment found 1.")
    elif has_activated:
        raise Exception("No virtual environment found 2.")


if __name__ == "__main__":
    # Temporarily redirect stdout for .env.enter eval
    # import io
    # sys.stdout = io.StringIO()

    current_dir = sys.argv[1]
    current_python_path = sys.argv[2]
    virtual_python_path = sys.argv[3]

    for folder in traverse_directory(
        base_dir=current_dir,
        includes=["<folder>/bin/activate"],
        excludes=["<folder>/node_modules"],
        limit=1,
        direction="forward",
        max_forward_depth=1,
    ):
        current_python_path = f"{folder}/bin/python"

    command = ''
    try:
        nearest_activation = activate_nearest(
            current_dir, current_python_path, virtual_python_path)
        command = f"source {nearest_activation}"
        # logger.success(f"Activated virtual environment in: {nearest_activation}")
    except Exception as e:
        if virtual_python_path:
            command = "deactivate"
        # logger.error(e)

    # Restore real stdout
    # sys.stdout = sys.__stdout__

    # logger.info(command)
    print(command)
