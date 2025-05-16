from jet.llm.mlx.tasks.relation_extraction import relation_extraction
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import ModelType

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    text: str = "Elon Musk, CEO of Tesla, announced a new factory opening in Shanghai."
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = relation_extraction(
            text=text,
            model_path=model,
            method=method,
            max_tokens=100
        )
        logger.log("Input Text:", text, colors=["GRAY", "DEBUG"])
        logger.log("Relations:", result["relations"], colors=[
                   "GRAY", "SUCCESS"])
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
