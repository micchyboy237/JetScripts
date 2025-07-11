from jet.code.markdown_utils._markdown_analyzer import link_to_text_ratio
from jet.code.markdown_utils._markdown_parser import base_parse_markdown
from jet.code.markdown_utils._preprocessors import preprocess_markdown
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard


input_text = "* * *\n* [ Reddit reReddit: Top posts of March 9, 2024\n* * * ](https://www.reddit.com/posts/2024/march-9-1/)\n* [ Reddit reReddit: Top posts of March 2024\n* * * ](https://www.reddit.com/posts/2024/march/)\n* [ Reddit reReddit: Top posts of 2024\n* * * ](https://www.reddit.com/posts/2024/)"
links = scrape_links(input_text)
# input_text = preprocess_markdown(input_text)
result = base_parse_markdown(input_text, ignore_links=False)
final = {
    "links": links,
    "result": result
}
copy_to_clipboard(final)
logger.success(format_json(result))
