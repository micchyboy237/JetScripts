import os
import shutil
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.utils.file import search_files

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == '__main__':
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/AI"
    includes = ["examples"]
    excludes = [".venv", ".pytest_cache", "node_modules"]
    extensions = [".py"]
    results = search_files(
        base_dir,
        extensions,
        include_files=includes,
        exclude_files=excludes,
    )
    logger.success(f"Results: ({len(results)})")

    save_file(results, f"{OUTPUT_DIR}/results.json")
