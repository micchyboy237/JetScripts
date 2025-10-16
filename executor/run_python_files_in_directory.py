from typing import Literal
from jet.executor.python_runner import run_python_files_in_directory
from jet.utils.file import search_files

def filter_files(base_dir):
    include_contents = ["*if __name__ == *"]
    excludes = []
    extensions = [".py"]
    results = search_files(
        base_dir,
        extensions,
        exclude_files=excludes,
        include_contents=include_contents,
    )
    return results

if __name__ == "__main__":
    target_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/stanza"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/libs/stanza/generated/all_files_with_main_block/runner_status"
    includes = filter_files(target_dir)
    excludes = [
        # "ambiguity-clarity.py",
    ]
    rerun_mode: Literal["all", "failed",
                        "unrun", "failed_and_unrun"] = "failed"
    run_python_files_in_directory(
        target_dir,
        includes=includes,
        excludes=excludes,
        output_dir=output_dir,
        rerun_mode=rerun_mode,
        recursive=True,
    )
