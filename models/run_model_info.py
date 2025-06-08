import os
from jet.file.utils import save_file
from jet.models.utils import get_model_info, get_model_limits
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "constants")

    model_info = get_model_info()
    logger.success(format_json(model_info))

    save_file(model_info["models"], f"{output_dir}/models.json")
    save_file(model_info["contexts"], f"{output_dir}/model_contexts.json")
    save_file(model_info["embeddings"], f"{output_dir}/model_embeddings.json")
    save_file(model_info["missing"], f"{output_dir}/missing.json")

    for model_id in model_info["missing"]:
        try:
            max_contexts, max_embeddings = get_model_limits(model_id)
            print(model_id, max_contexts, max_embeddings)
        except:
            continue
