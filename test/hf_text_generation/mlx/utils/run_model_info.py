from jet.llm.mlx.models import get_model_info, resolve_model
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    model_info = get_model_info()
    logger.success(format_json(model_info))
