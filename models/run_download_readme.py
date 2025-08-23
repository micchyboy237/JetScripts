from jet.models.extract_hf_readme_code import extract_code_from_hf_readmes
from jet.models.model_types import ModelKey, ModelType, ModelValue
from jet.models.utils import download_model_readmes, download_readme, resolve_model_key, resolve_model_value

if __name__ == "__main__":
    overwrite = True
    extract_code = True
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    model: ModelType = "jinaai/jina-embeddings-v2-base-en"

    model_value: ModelValue = resolve_model_value(model)
    download_readme(model_value, hf_readme_download_dir,
                    overwrite, extract_code)
