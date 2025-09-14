import os
import shutil
from jet.search.firecrawl import search_web
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
    scrape = True
    search_results = search_web(query, scrape=scrape)
    logger.gray("\nSearch Results:")
    logger.success(search_results)
    save_file(search_results, f"{OUTPUT_DIR}/search_results.json")
