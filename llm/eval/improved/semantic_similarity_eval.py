# !pip install llama-index
import asyncio
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.base.embeddings.base import SimilarityMode

from jet.llm.ollama.base import create_embed_model
from jet.logger import logger


async def main():
    logger.newline()
    logger.debug("Preparing ollama evaluator...")
    evaluator = SemanticSimilarityEvaluator(
        embed_model=create_embed_model()
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
    result = await evaluator.aevaluate(
        response=response,
        reference=reference,
        similarity_threshold=0.8,
    )

    logger.log("Score:", result.score, colors=["LOG", "SUCCESS"])
    # default similarity threshold is 0.8
    logger.log("Passing:", result.passing, colors=["LOG", "SUCCESS"])

    response = "Sorry, I do not have sufficient context to answer this question."
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
    result = await evaluator.aevaluate(
        response=response,
        reference=reference,
        similarity_threshold=0.8,
    )

    logger.log("Score:", result.score, colors=["LOG", "SUCCESS"])
    # default similarity threshold is 0.8
    logger.log("Passing:", result.passing, colors=["LOG", "SUCCESS"])

    local_embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model = resolve_embed_model(f"local:{local_embed_model}")

    logger.newline()
    logger.debug("Preparing sentence transformer evaluator...")
    evaluator = SemanticSimilarityEvaluator(
        embed_model=embed_model,
        similarity_mode=SimilarityMode.DEFAULT,
        similarity_threshold=0.6,
    )
    response = "The sky is yellow."
    reference = "The sky is blue."

    logger.newline()
    logger.log("Embed Model:", local_embed_model, colors=["GRAY", "INFO"])
    logger.log("Response:", response, colors=["LOG", "DEBUG"])
    logger.log("Reference:", reference, colors=["LOG", "DEBUG"])
    logger.debug("Evaluating...")
    result = await evaluator.aevaluate(
        response=response,
        reference=reference,
    )

    logger.log("Score:", result.score, colors=["LOG", "SUCCESS"])
    # default similarity threshold is 0.6
    logger.log("Passing:", result.passing, colors=["LOG", "SUCCESS"])

# Call the main function
if __name__ == "__main__":  # Add this block
    asyncio.run(main())  # Run the async main function
