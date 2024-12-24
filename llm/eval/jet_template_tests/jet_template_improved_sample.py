# !pip install llama-index
import os
import json
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from jet.llm.ollama import (
    update_llm_settings,
    large_llm_model,
    large_embed_model,
)
from jet.transformers import make_serializable
from jet.logger import logger


def load_llm_settings():
    logger.newline()
    logger.debug("Loading LLM settings...")
    settings = update_llm_settings({
        "llm_model": large_llm_model,
        "embedding_model": large_embed_model,
    })
    return settings


def save_results(data, base_dir, file_name):
    os.makedirs(base_dir, exist_ok=True)
    results_path = os.path.join(base_dir, file_name)
    with open(results_path, "w") as f:
        json.dump(make_serializable(data),
                  f, indent=2, ensure_ascii=False)
        logger.log("Saved to", results_path, colors=[
            "WHITE", "BRIGHT_SUCCESS"])


async def main():
    settings = update_llm_settings()
    logger.newline()
    logger.debug("Preparing ollama evaluator...")
    evaluator = SemanticSimilarityEvaluator(
        embed_model=settings.embed_model
    )

    response = "The sky is typically blue"
    reference = """The color of the sky can vary depending on several factors, including time of day, weather conditions, and location.

    During the day, when the sun is in the sky, the sky often appears blue. 
    This is because of a phenomenon called Rayleigh scattering, where molecules and particles in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves. 
    This is why we perceive the sky as blue on a clear day.
    """

    logger.newline()
    logger.log("Embed Model:", evaluator._embed_model.model_name,
               colors=["GRAY", "INFO"])
    logger.log("Response:", response, colors=["LOG", "DEBUG"])
    logger.log("Reference:", reference, colors=["LOG", "DEBUG"])
    logger.debug("Evaluating...")
    result = evaluator.evaluate(
        response=response,
        reference=reference,
        similarity_threshold=0.8,
    )
    logger.log("Score:", result.score, colors=["LOG", "SUCCESS"])
    logger.log("Passing:", result.passing, colors=["LOG", "SUCCESS"])

    base_dir = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/improved"
    save_results(result, base_dir)


if __name__ == "__main__":
    main()
