from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings

initialize_ollama_settings()

"""
# How to select examples by maximal marginal relevance (MMR)

The `MaxMarginalRelevanceExampleSelector` selects [examples](/docs/concepts/example_selectors/) based on a combination of which examples are most similar to the inputs, while also optimizing for diversity. It does this by finding the examples with the embeddings that have the greatest cosine similarity with the inputs, and then iteratively adding them while penalizing them for closeness to already selected examples.
"""


example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OllamaEmbeddings(model="nomic-embed-text"),
    FAISS,
    k=2,
)
mmr_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

logger.success(mmr_prompt.format(adjective="worried"))

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OllamaEmbeddings(model="nomic-embed-text"),
    FAISS,
    k=2,
)
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
logger.success(similar_prompt.format(adjective="worried"))


logger.info("\n\n[DONE]", bright=True)
