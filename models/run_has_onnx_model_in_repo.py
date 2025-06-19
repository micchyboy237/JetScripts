from jet.logger import logger
from jet.models.onnx_model_checker import has_onnx_model_in_repo


if __name__ == "__main__":
    repo_id = "static-retrieval-mrl-en-v1"
    result = has_onnx_model_in_repo(repo_id)
    logger.success(
        f"\nONNX model (standard or ARM64) exists in {repo_id}: {result}")
