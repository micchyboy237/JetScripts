from jet.llm.mlx.models import resolve_model
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    logger.gray("\nResolve LLM model")
    model = "qwen3-4b-3bit"
    logger.debug(model)
    logger.success(resolve_model(model))

    logger.gray("\nResolve embed model")
    embed_model = "all-minilm-l6-v2-8bit"
    logger.debug(embed_model)
    logger.success(resolve_model(embed_model))
