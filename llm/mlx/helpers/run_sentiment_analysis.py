from typing import List
from jet.llm.mlx.helpers.sentiment_analysis import sentiment_analysis
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    text: str = "I absolutely love this product! It's amazing."
    sentiments: List[str] = ["Positive", "Negative", "Neutral"]
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = sentiment_analysis(
            text=text,
            sentiments=sentiments,
            model_path=model,
            method=method
        )
        logger.log("Text:", text, colors=["GRAY", "DEBUG"])
        logger.log("Sentiment:", result["sentiment"], colors=[
                   "GRAY", "SUCCESS"])
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
