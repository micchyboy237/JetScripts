from jet.llm.mlx.helpers.text_completion import text_completion
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import ModelType

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    prompt: str = "Once upon a time, in a land far away, there was a"
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = text_completion(
            prompt=prompt,
            model_path=model,
            method=method,
            max_tokens=50
        )
        logger.log("Prompt:", prompt, colors=["GRAY", "DEBUG"])
        logger.log("Completion:", result["completion"], colors=["GRAY", "SUCCESS"])
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