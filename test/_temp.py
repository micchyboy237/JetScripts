from jet.search.playwright.playwright_search import PlaywrightSearch
from jet.transformers.formatters import format_json
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

query = "Top 10 isekai anime 2025 with release date, synopsis, number of episode, airing status"

searcher = PlaywrightSearch(max_results=10, topic="general")
result = searcher._run(query=query)
logger.gray("Synchronous search result:")
logger.success(format_json(result))
save_file(result, f"{OUTPUT_DIR}/sync_result.json")