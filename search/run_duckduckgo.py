# JetScripts/search/run_duckduckgo.py
import os
import shutil
from jet.search.duckduckgo import DuckDuckGoSearch
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    with DuckDuckGoSearch() as search:
        query = "Top isekai anime 2025"

        # Text search
        results = search.text(query, max_results=10)
        logger.gray("\nText sample")
        logger.success(format_json(results[0]))
        save_file(results, f"{OUTPUT_DIR}/text_results.json")

        # News search
        results = search.news(query=query, timelimit="d", max_results=3)
        logger.gray("\nNews sample")
        logger.success(format_json(results[0]))
        save_file(results, f"{OUTPUT_DIR}/news_results.json")

        # Image search
        results = search.images(
            query=query, region="uk-en", safesearch="off", max_results=5)
        logger.gray("\nImage sample")
        logger.success(format_json(results[0]))
        save_file(results, f"{OUTPUT_DIR}/image_results.json")

        # Video search
        results = search.videos(query=query, max_results=4)
        logger.gray("\nVideo sample")
        logger.success(format_json(results[0]))
        save_file(results, f"{OUTPUT_DIR}/video_results.json")
