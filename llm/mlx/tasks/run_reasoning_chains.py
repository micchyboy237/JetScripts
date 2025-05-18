from jet.llm.mlx.tasks.reasoning_chains import reasoning_chains
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import LLMModelType

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    problem: str = "If a car travels 60 miles in 2 hours, what is its average speed?"
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = reasoning_chains(
            problem=problem,
            model_path=model,
            method=method,
            max_tokens=150
        )
        logger.log("Problem:", problem, colors=["GRAY", "DEBUG"])
        logger.log("Reasoning:", result["reasoning"], colors=[
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
