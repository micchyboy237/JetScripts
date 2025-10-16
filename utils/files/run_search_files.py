import os
import shutil
from jet.file.utils import save_file
from jet.logger import logger
from jet.utils.file import search_files

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == '__main__':
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/stanza"
    includes = []
    include_contents = ["*if __name__ == *"]
    excludes = []
    extensions = [".py"]
    results = search_files(
        base_dir,
        extensions,
        exclude_files=excludes,
        include_contents=include_contents,
    )
    logger.success(f"Results: ({len(results)})")

    save_file(results, f"{OUTPUT_DIR}/results.json")
