import os
import shutil
from jet.file.utils import save_file
from jet.logger import logger
from jet.utils.file import search_files

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == '__main__':
    # base_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/examples/Context-Engineering"
    base_dir = [
        # "/Users/jethroestrada/.cache/huggingface/hub",
        # "/Users/jethroestrada/.cache/huggingface/datasets",
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/silero-vad",
    ]
    # includes = []
    includes = [
        # "*japanese*",
        # "notebook",
        # "tutorial",
        "test_*",
    ]
    excludes = [
        # ".locks"
    ]
    # include_contents = ["*__main__*"]
    include_contents = []
    exclude_contents = []
    # extensions = []
    extensions = [".py", ".ipynb"]
    results = search_files(
        base_dir,
        extensions,
        include_files=includes,
        exclude_files=excludes,
        include_contents=include_contents,
        exclude_contents=exclude_contents,
    )
    logger.success(f"Results: ({len(results)})")

    save_file(results, f"{OUTPUT_DIR}/results.json")
