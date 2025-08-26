import os
import shutil
from jet.executor.python_runner import run_python_files_in_directory
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
# shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    includes = []
    # excludes = []

    # Gather all file names (without extension) under OUTPUT_DIR/failed and OUTPUT_DIR/success recursively
    excludes = []
    for status_dir in ["failed", "success"]:
        dir_path = os.path.join(OUTPUT_DIR, status_dir)
        if os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".log"):
                        # Remove .log extension
                        file_name = os.path.splitext(file)[0]
                        excludes.append(file_name)
    if excludes:
        logger.info("\nFiles to exclude")
        logger.debug(format_json(excludes, indent=1))

    run_python_files_in_directory(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/swarms/examples",
        includes=includes,
        excludes=excludes,
        output_dir=OUTPUT_DIR,
        recursive=True,
    )
