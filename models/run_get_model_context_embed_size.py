import argparse
import os

from jet.file.utils import save_file
from jet.logger import logger
from jet.models.utils import get_model_limits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lookup and save the maximum context and embedding size for a model."
    )
    parser.add_argument(
        "model_id",
        type=str,
        nargs="?",
        default="Qwen/Qwen3.5-2B",
        help="Model HuggingFace ID (default: 'Qwen/Qwen3.5-2B')",
    )

    args = parser.parse_args()
    output_dir = os.path.join(os.path.dirname(__file__), "constants")
    model_id = args.model_id

    max_context, max_embeddings = get_model_limits(model_id)

    logger.newline()
    logger.log("Model ID:", f" {model_id}", colors=["GRAY", "DEBUG"])
    logger.log("Max Context:", f" {max_context}", colors=["GRAY", "SUCCESS"])
    logger.log("Max Embeddings:", f" {max_embeddings}", colors=["GRAY", "SUCCESS"])

    save_file(
        {
            "model_id": model_id,
            "max_context": max_context,
            "max_embeddings": max_embeddings,
        },
        f"{output_dir}/model_context_embed_size.json",
    )
