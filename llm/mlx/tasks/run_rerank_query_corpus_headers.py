from jet.llm.mlx.tasks.rerank_query_corpus_headers import rerank_query_corpus_headers
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

query = "What is the capital of France?"
corpus = [
    {"title": "France Overview",
        "content": "France is a country in Europe with Paris as its capital."},
    {"title": "Florida Guide", "content": "Florida is a state in the USA."},
    {"title": "Paris Highlights",
        "content": "Paris is a major city known for the Eiffel Tower."}
]
headers = ["title", "content"]
model: ModelType = "llama-3.2-3b-instruct-4bit"

result = rerank_query_corpus_headers(
    query, corpus, headers, model, max_tokens=10)
logger.gray("Result:")
logger.success(format_json(result))
