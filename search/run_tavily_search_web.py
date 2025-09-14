import os
import shutil
from jet.search.tavily import tavily_search
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    query = "Top 10 isekai anime 2025 with release date, synopsis, number of episode, airing status"
    response_format = "json"
    search_results = tavily_search(query, response_format)
    logger.gray("\nSearch Results:")
    logger.success(search_results)
    # Save results to file based on format
    file_extension = "md" if response_format == "markdown" else "json"
    save_file(search_results, f"{OUTPUT_DIR}/search_results.{file_extension}")
