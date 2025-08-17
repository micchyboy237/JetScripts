from jet.llm.mlx.tasks.keyword_extraction import keyword_extraction
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.models.model_types import LLMModelType

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    text: str = "The Amazon rainforest, located in South America, is the world's largest tropical rainforest. It is home to millions of species and plays a critical role in global climate regulation."
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = keyword_extraction(
            text=text,
            model_path=model,
            method=method,
            max_tokens=50
        )
        logger.log("Input Text:", text, colors=["GRAY", "DEBUG"])
        logger.log("Keywords:", result["keywords"], colors=["GRAY", "SUCCESS"])
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
