from jet.llm.mlx.helpers.code_and_output import code_and_output
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import ModelType

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    description: str = "Write a Python function to calculate the factorial of a number."
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = code_and_output(
            description=description,
            model_path=model,
            method=method,
            max_tokens=150
        )
        logger.log("Description:", description, colors=["GRAY", "DEBUG"])
        logger.log("Code:", result["code"], colors=["GRAY", "SUCCESS"])
        logger.log("Expected Output:", result["expected_output"], colors=["GRAY", "SUCCESS"])
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