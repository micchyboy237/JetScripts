import argparse
import os

from jet.adapters.llama_cpp.model_utils import get_all_models_ctx_embd_sizes
from jet.file.utils import save_file
from jet.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lookup and save the maximum context and embedding size for a model."
    )
    parser.add_argument(
        "model_id",
        type=str,
        nargs="?",
        default="google/embeddinggemma-300m",
        help="Model HuggingFace ID (ex: 'Qwen/Qwen3.5-2B')",
    )

    args = parser.parse_args()
    output_dir = os.path.join(os.path.dirname(__file__), "constants")
    model_id = args.model_id

    models_ctx_embd_sizes = get_all_models_ctx_embd_sizes()

    logger.newline()
    logger.log("Models:", f" {len(models_ctx_embd_sizes)}", colors=["GRAY", "DEBUG"])

    save_file(models_ctx_embd_sizes, f"{output_dir}/model_context_embed_size.json")
