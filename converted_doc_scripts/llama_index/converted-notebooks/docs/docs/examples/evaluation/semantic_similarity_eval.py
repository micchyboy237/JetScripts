import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.embeddings import SimilarityMode, resolve_embed_model
from llama_index.core.evaluation import SemanticSimilarityEvaluator
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/semantic_similarity_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Embedding Similarity Evaluator

This notebook shows the `SemanticSimilarityEvaluator`, which evaluates the quality of a question answering system via semantic similarity.

Concretely, it calculates the similarity score between embeddings of the generated answer and the reference answer.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Embedding Similarity Evaluator")

# !pip install llama-index


evaluator = SemanticSimilarityEvaluator()

response = "The sky is typically blue"
reference = """The color of the sky can vary depending on several factors, including time of day, weather conditions, and location.

During the day, when the sun is in the sky, the sky often appears blue.
This is because of a phenomenon called Rayleigh scattering, where molecules and particles in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves.
This is why we perceive the sky as blue on a clear day.
"""

async def async_func_14():
    result = evaluator.evaluate(
        response=response,
        reference=reference,
    )
    return result
result = asyncio.run(async_func_14())
logger.success(format_json(result))

logger.debug("Score: ", result.score)
logger.debug("Passing: ", result.passing)  # default similarity threshold is 0.8

response = "Sorry, I do not have sufficient context to answer this question."
reference = """The color of the sky can vary depending on several factors, including time of day, weather conditions, and location.

During the day, when the sun is in the sky, the sky often appears blue.
This is because of a phenomenon called Rayleigh scattering, where molecules and particles in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves.
This is why we perceive the sky as blue on a clear day.
"""

async def async_func_30():
    result = evaluator.evaluate(
        response=response,
        reference=reference,
    )
    return result
result = asyncio.run(async_func_30())
logger.success(format_json(result))

logger.debug("Score: ", result.score)
logger.debug("Passing: ", result.passing)  # default similarity threshold is 0.8

"""
### Customization
"""
logger.info("### Customization")


embed_model = resolve_embed_model("local")
evaluator = SemanticSimilarityEvaluator(
    embed_model=embed_model,
    similarity_mode=SimilarityMode.DEFAULT,
    similarity_threshold=0.6,
)

response = "The sky is yellow."
reference = "The sky is blue."

async def async_func_13():
    result = evaluator.evaluate(
        response=response,
        reference=reference,
    )
    return result
result = asyncio.run(async_func_13())
logger.success(format_json(result))

logger.debug("Score: ", result.score)
logger.debug("Passing: ", result.passing)

"""
We note here that a high score does not imply the answer is always correct.  

Embedding similarity primarily captures the notion of "relevancy". Since both the response and reference discuss "the sky" and colors, they are semantically similar.
"""
logger.info("We note here that a high score does not imply the answer is always correct.")

logger.info("\n\n[DONE]", bright=True)