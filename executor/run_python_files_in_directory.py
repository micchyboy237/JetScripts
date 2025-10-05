from typing import Literal
from jet.executor.python_runner import run_python_files_in_directory


if __name__ == "__main__":
    target_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/bertopic/jet_examples/more_usage_examples"
    output_dir = f"{target_dir}/generated/runner_status"
    includes = [
        # "file_name.py",
    ]
    excludes = [
        # "ambiguity-clarity.py",
    ]
    rerun_mode: Literal["all", "failed",
                        "unrun", "failed_and_unrun"] = "all"
    run_python_files_in_directory(
        target_dir,
        includes=includes,
        excludes=excludes,
        output_dir=output_dir,
        rerun_mode=rerun_mode,
    )
