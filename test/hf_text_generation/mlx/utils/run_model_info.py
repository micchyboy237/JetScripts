from jet.llm.mlx.models import get_model_info
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard

if __name__ == "__main__":
    model_info = get_model_info()
    logger.success(format_json(model_info))
    copy_to_clipboard(model_info)
