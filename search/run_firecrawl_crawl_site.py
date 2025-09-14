import os
import shutil
from jet.search.firecrawl import crawl_entire_site_firecrawl, search_web
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    url = "https://docs.llamaindex.ai/en/stable"

    crawled_content = crawl_entire_site_firecrawl(
        url,
        limit=20,
        formats=["markdown"],
        max_wait_time=600,
    )
    print(crawled_content)

    logger.gray("\nCrawler Content:")
    logger.success(crawled_content)
    save_file(crawled_content, f"{OUTPUT_DIR}/crawled_content.md")
