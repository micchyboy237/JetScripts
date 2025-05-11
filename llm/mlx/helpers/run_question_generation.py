from jet.llm.mlx.helpers.question_generation import question_generation
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import ModelType

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    text: str = "The Eiffel Tower, located in Paris, France, was completed in 1889. It was designed by Gustave Eiffel and serves as a global cultural icon."
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = question_generation(
            text=text,
            model_path=model,
            method=method,
            max_tokens=100
        )
        logger.log("Input Text:", text, colors=["GRAY", "DEBUG"])
        logger.log("Questions:", result["questions"], colors=["GRAY", "SUCCESS"])
        logger.log("Token IDs:", result['token_ids'], colors=["GRAY", "DEBUG"])
        logger.log("Valid:", result['is_valid'], colors=[
                   "GRAY", "SUCCESS" if result['is_valid'] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result['error'], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", colors=["GRAY", "SUCCESS"])
        logger.newline()
        logger.gray("Result:")
        logger.success(format_json(result))