from jet.llm.mlx.tasks.rerank_query_corpus import rerank_query_corpus
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

query = "What is the capital of France?"
corpus = [
    "France is a country in Europe with Paris as its capital.",
    "Florida is a state in the USA.",
    "Paris is a major city known for the Eiffel Tower."
]
model: LLMModelType = "llama-3.2-3b-instruct-4bit"

result = rerank_query_corpus(query, corpus, model, max_tokens=10)
logger.gray("Result:")
logger.success(format_json(result))
