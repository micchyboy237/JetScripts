from jet.llm.mlx.tasks.text_classification import text_classification
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import LLMModelType
from typing import List

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    text: str = "Win a free iPhone now! Click here to claim your prize!"
    labels: List[str] = ["Spam", "Not Spam"]
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = text_classification(
            text=text,
            labels=labels,
            model_path=model,
            method=method
        )
        logger.log("Text:", text, colors=["GRAY", "DEBUG"])
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
