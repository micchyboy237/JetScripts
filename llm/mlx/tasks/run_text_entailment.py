from jet.llm.mlx.tasks.text_entailment import text_entailment
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import ModelType
from typing import List

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    premise: str = "All birds can fly."
    hypothesis: str = "Penguins can fly."
    labels: List[str] = ["Entailment", "Contradiction", "Neutral"]
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = text_entailment(
            premise=premise,
            hypothesis=hypothesis,
            labels=labels,
            model_path=model,
            method=method
        )
        logger.log("Premise:", premise, colors=["GRAY", "DEBUG"])
        logger.log("Hypothesis:", hypothesis, colors=["GRAY", "DEBUG"])
        logger.log("Label:", result["label"], colors=["GRAY", "SUCCESS"])
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
