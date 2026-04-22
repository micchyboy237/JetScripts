import argparse

from jet.models.model_types import ModelType, ModelValue
from jet.models.utils import download_readme, resolve_model_value


def main():
    parser = argparse.ArgumentParser(
        description="Download and process model README files."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name or path (e.g., 'speechbrain/vad-crdnn-libriparty')",
    )
    args = parser.parse_args()

    overwrite = True
    extract_code = True
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    model: ModelType = args.model

    model_value: ModelValue = resolve_model_value(model)
    download_readme(model_value, hf_readme_download_dir, overwrite, extract_code)


if __name__ == "__main__":
    main()
