from jet.llm.mlx.helpers.anaphora_resolution import AnaphoraResolutionResult, anaphora_resolution
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    input_text: str = (
        "John went to the store. He bought some apples. They were fresh."
    )

    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result: AnaphoraResolutionResult = anaphora_resolution(
            input_text, model, method=method
        )
        logger.log("Input Text:", input_text, colors=["GRAY", "DEBUG"])
        logger.log("Resolved Text:", result["resolved_text"], colors=["GRAY", "SUCCESS"])
        logger.log("Valid:", result["is_valid"], colors=[
                   "GRAY", "SUCCESS" if result["is_valid"] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result["error"], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", method, colors=["GRAY", "SUCCESS"])
        logger.newline()