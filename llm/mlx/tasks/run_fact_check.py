from typing import List
from jet.llm.mlx.tasks.fact_check import fact_check
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    statement: str = "The Earth is flat."
    verdicts: List[str] = ["True", "False", "Uncertain"]
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = fact_check(
            statement=statement,
            verdicts=verdicts,
            model_path=model,
            method=method
        )
        logger.log("Statement:", statement, colors=["GRAY", "DEBUG"])
        logger.log("Verdict:", result["verdict"], colors=["GRAY", "SUCCESS"])
        logger.log("Token ID:", result['token_id'], colors=["GRAY", "DEBUG"])
        logger.log("Valid:", result['is_valid'], colors=[
                   "GRAY", "SUCCESS" if result['is_valid'] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result['error'], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", colors=["GRAY", "SUCCESS"])
        logger.newline()
        logger.gray("Result:")
        logger.success(format_json(result))
