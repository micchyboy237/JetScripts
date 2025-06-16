from jet.models.onnx_model_checker import has_onnx_model_in_repo


if __name__ == "__main__":
    repo_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    result = has_onnx_model_in_repo(repo_id)
    print(f"ONNX model (standard or ARM64) exists in {repo_id}: {result}")
