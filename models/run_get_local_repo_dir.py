import os
from jet.models.utils import get_local_repo_dir
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    model_id = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
    cache_dir = get_local_repo_dir(model_id)
    logger.success(f"Cache dir: {cache_dir}")
