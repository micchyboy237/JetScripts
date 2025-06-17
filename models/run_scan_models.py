import os
from jet.file.utils import save_file
from jet.models.utils import scan_local_hf_models


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "constants")

    overwrite = False
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    # Download MLX Model README.md files.
    local_models = scan_local_hf_models()

    save_file(local_models, f"{output_dir}/model_paths.json")
