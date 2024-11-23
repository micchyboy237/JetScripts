import os
import logging
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def main():
    # Example model name
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Optionally set a custom cache directory for SentenceTransformers
    # os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/path/to/your/custom/cache"

    # Call the get_model_path function with update_model flag
    model_path = get_model_path(model_name, update_model=True)
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
