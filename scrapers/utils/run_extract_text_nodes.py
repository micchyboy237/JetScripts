import os
import shutil

from jet.file.utils import save_file
from jet.scrapers.text_nodes import extract_text_nodes

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    link = "https://docs.tavily.com/documentation/api-reference/endpoint/crawl"

    text_nodes = extract_text_nodes(link)
    save_file(text_nodes, f"{OUTPUT_DIR}/text_nodes.json")

    text_elements = [dict(node.get_element_details()) for node in text_nodes]
    save_file(text_elements, f"{OUTPUT_DIR}/text_elements.json")
