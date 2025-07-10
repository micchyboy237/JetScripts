from jet.code.markdown_utils._markdown_analyzer import link_to_text_ratio
from jet.logger import logger
from jet.transformers.formatters import format_json


input_text = "* * *\n* [ Reddit reReddit: Top posts of March 9, 2024\n* * * ](https://www.reddit.com/posts/2024/march-9-1/)\n* [ Reddit reReddit: Top posts of March 2024\n* * * ](https://www.reddit.com/posts/2024/march/)\n* [ Reddit reReddit: Top posts of 2024\n* * * ](https://www.reddit.com/posts/2024/)"
result = link_to_text_ratio(input_text, threshold=0.3)
logger.success(format_json(result))
