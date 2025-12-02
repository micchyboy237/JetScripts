from jet.models.model_types import ModelType, ModelValue
from jet.models.utils import download_readme, resolve_model_value

if __name__ == "__main__":
    overwrite = True
    extract_code = True
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    model: ModelType = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"

    model_value: ModelValue = resolve_model_value(model)
    download_readme(model_value, hf_readme_download_dir,
                    overwrite, extract_code)
