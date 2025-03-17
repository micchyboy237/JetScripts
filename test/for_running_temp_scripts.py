from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.n_grams import count_ngrams, get_most_common_ngrams

request_queries = [
    "React Native",
    "Mobile development"
]


query_ngrams = [list(get_most_common_ngrams(
    query, min_count=1, max_words=5)) for query in request_queries]
results = ["_".join(text.split())
           for queries in query_ngrams for text in queries]

copy_to_clipboard(results)
logger.success(format_json(results))
