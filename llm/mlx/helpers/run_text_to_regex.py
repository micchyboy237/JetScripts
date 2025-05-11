from jet.llm.mlx.helpers.text_to_regex import TextToRegexResult, text_to_regex
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    input_text: str = "Match any email address ending with '.com'."

    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result: TextToRegexResult = text_to_regex(
            input_text, model, method=method
        )
        logger.log("Input Text:", input_text, colors=["GRAY", "DEBUG"])
        logger.log("Regex:", result["regex"], colors=["GRAY", "SUCCESS"])
        logger.log("Valid:", result["is_valid"], colors=[
                   "GRAY", "SUCCESS" if result["is_valid"] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result["error"], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", method, colors=["GRAY", "SUCCESS"])
        logger.newline()