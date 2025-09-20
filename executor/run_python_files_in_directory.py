import os
from typing import Literal
from jet.executor.python_runner import run_python_files_in_directory


if __name__ == "__main__":
    target_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/Prompt_Engineering/converted-notebooks/all_prompt_engineering_techniques"
    output_dir = f"{target_dir}/generated/runner_status"
    includes = [
        # "file_name.py",
    ]
    excludes = [
        "ambiguity-clarity.py",
        "basic-prompt-structures.py",
        "constrained-guided-generation.py",
        "cot-prompting.py",
        "ethical-prompt-engineering.py",
        "evaluating-prompt-effectiveness.py",
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
