# JetScripts/search/run_duckduckgo.py
import os
import shutil
from jet.search.duckduckgo import search_web
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    # search = DuckDuckGoSearch()

    query = "Top isekai anime 2025"
    search_results = search_web(query)
    logger.gray("\nSearch Results:")
    logger.success(search_results)
    save_file(search_results, f"{OUTPUT_DIR}/search_results.txt")
