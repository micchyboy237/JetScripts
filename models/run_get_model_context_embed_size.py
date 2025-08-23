import os
from jet.file.utils import save_file
from jet.models.model_types import ModelType
from jet.models.utils import get_model_info, get_model_limits
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "constants")
    model_id: ModelType = "jinaai/jina-embeddings-v2-base-en"

    max_context, max_embeddings = get_model_limits(model_id)

    logger.newline()
    logger.log("Model ID:", f" {model_id}", colors=["GRAY", "DEBUG"])
    logger.log("Max Context:", f" {max_context}", colors=["GRAY", "SUCCESS"])
    logger.log("Max Embeddings:", f" {max_embeddings}", colors=[
               "GRAY", "SUCCESS"])

    save_file({
        "model_id": model_id,
        "max_context": max_context,
        "max_embeddings": max_embeddings,
    }, f"{output_dir}/model_context_embed_size.json")
