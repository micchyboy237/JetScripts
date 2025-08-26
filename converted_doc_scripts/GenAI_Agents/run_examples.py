import os
import shutil
from jet.executor.python_runner import run_python_files_in_directory

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    includes = []
    excludes = []
    run_python_files_in_directory(
        os.path.join(os.path.dirname(__file__), "converted-notebooks"),
        includes=includes,
        excludes=excludes,
        output_dir=OUTPUT_DIR,
        recursive=True,
    )
