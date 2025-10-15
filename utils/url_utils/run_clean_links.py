import os

from jet.utils.url_utils import clean_links
from jet.file.utils import save_file
from jet.logger import logger

text_with_sample_links = """
Links: https://site1.com/page1?x=1 and https://site2.com/path/#anchor
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    result = clean_links(text_with_sample_links)
    
    logger.debug(f"\nText: '{text_with_sample_links}'")
    logger.success(f"\nResult: '{result}'")

    save_file(result, f"{output_dir}/result.txt")
