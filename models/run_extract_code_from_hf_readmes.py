import os
import re
import shutil

from jet.models.extract_hf_readme_code import extract_code_from_hf_readmes
from jet.models.utils import download_model_readmes


def main():
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"

    # output_dir = os.path.join(os.path.dirname(__file__), "hf_extracted_code")
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes/code"

    include_text = True  # Set to True to include non-code text
    extract_code_from_hf_readmes(
        hf_readme_download_dir, output_dir, include_text)


if __name__ == "__main__":
    main()
