import os
import shutil
from pathlib import Path
from jet.models.extract_hf_readme_code import extract_code_from_hf_readmes
from jet.models.utils import download_model_readmes, resolve_model_key
from jet.models.constants import MODEL_VALUES_LIST


def main():
    overwrite = True
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    output_dir = f"{hf_readme_download_dir}/code"

    # Download READMEs for all models
    download_model_readmes(hf_readme_download_dir, overwrite)


if __name__ == "__main__":
    main()
