from jet.llm.mlx.tasks.translation_pairs import TranslationResult, translation_pairs
from jet.models.model_types import LLMModelType
from jet.logger import logger

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    input_text: str = "The sun sets slowly behind the mountain."
    target_language: str = "Spanish"

    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result: TranslationResult = translation_pairs(
            input_text, target_language, model, method=method
        )
        logger.log("Input Text:", input_text, colors=["GRAY", "DEBUG"])
        logger.log("Translation:", result["translation"], colors=[
                   "GRAY", "SUCCESS"])
        logger.log("Valid:", result["is_valid"], colors=[
                   "GRAY", "SUCCESS" if result["is_valid"] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result["error"], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", method, colors=["GRAY", "SUCCESS"])
        logger.newline()
