import os
import shutil
from jet.executor.python_runner import run_python_files_in_directory


if __name__ == "__main__":
    target_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques"
    output_dir = f"{target_dir}/generated/runner_status"
    shutil.rmtree(output_dir, ignore_errors=True)
    includes = []
    excludes = []
    run_python_files_in_directory(
        target_dir,
        includes=includes,
        excludes=excludes,
        output_dir=output_dir,
    )
