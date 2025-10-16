from typing import Literal
from jet.executor.python_runner import run_python_files_in_directory

if __name__ == "__main__":
    target_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/stanza"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/libs/stanza/generated/all_files_with_main_block/runner_status"

    includes = []
    excludes = []
    include_contents = ["*if __name__ == *"]
    exclude_contents = ["*argparse*"]
    rerun_mode: Literal["all", "failed",
                        "unrun", "failed_and_unrun"] = "all"

    run_python_files_in_directory(
        target_dir,
        includes=includes,
        excludes=excludes,
        include_contents=include_contents,
        exclude_contents=exclude_contents,
        output_dir=output_dir,
        rerun_mode=rerun_mode,
        recursive=True,
    )
