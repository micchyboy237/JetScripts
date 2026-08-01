import os
import shutil
from typing import List

from jet.file.utils import save_file
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    link = "https://docs.tavily.com/documentation/api-reference/endpoint/crawl"

    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(link)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")
