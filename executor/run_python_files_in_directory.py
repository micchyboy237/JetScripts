import os
from typing import Literal
from jet.executor.python_runner import run_python_files_in_directory


if __name__ == "__main__":
    target_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques"
    output_dir = f"{target_dir}/generated/runner_status"
    includes = [
        # "17_graph_rag.py",
        "18_hierarchy_rag.py"
    ]
    excludes = []
    rerun_mode: Literal["all", "failed",
                        "unrun", "failed_and_unrun"] = "failed"
    run_python_files_in_directory(
        target_dir,
        includes=includes,
        excludes=excludes,
        output_dir=output_dir,
        rerun_mode=rerun_mode,
    )
