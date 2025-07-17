from jet.models.model_types import ModelKey, ModelType, ModelValue
from jet.models.utils import download_model_readmes, download_readme, resolve_model_key, resolve_model_value


if __name__ == "__main__":
    overwrite = True
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"

    model: ModelType = "nomic-embed-text"
    model_key: ModelKey = resolve_model_key(model)
    model_value: ModelValue = resolve_model_value(model)
    # Download MLX Model README.md files.
    download_readme(model_value, model_key, hf_readme_download_dir, overwrite)
