from jet.llm.mlx.tasks.instruction_response import instruction_response
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import LLMModelType

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    instruction: str = "Write a short poem about the moon."
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = instruction_response(
            instruction=instruction,
            model_path=model,
            method=method,
            max_tokens=100
        )
        logger.log("Instruction:", instruction, colors=["GRAY", "DEBUG"])
        logger.log("Response:", result["response"], colors=["GRAY", "SUCCESS"])
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
