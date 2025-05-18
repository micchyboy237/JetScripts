from pathlib import Path
import re
from typing import List
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard


def get_python_filenames(directory_path: str) -> List[str]:
    """
    Returns the base names of .py files in the specified directory (non-recursively).

    :param directory_path: Path to the directory
    :return: List of .py file names
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise ValueError(f"{directory_path} is not a valid directory")

    return [item.name for item in dir_path.iterdir()
            if item.is_file() and item.suffix == ".py"]


def sort_key(filename):
    match = re.match(r'^(\d+)', filename)
    return (int(match.group(1)) if match else float('inf'), filename)


python_files = get_python_filenames(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques")


sorted_files = sorted(python_files, key=sort_key)

logger.success(format_json(sorted_files))
copy_to_clipboard(format_json(sorted_files))
