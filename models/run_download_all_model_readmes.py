from jet.models.utils import download_model_readmes


if __name__ == "__main__":
    overwrite = False
    hf_readme_download_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    # Download MLX Model README.md files.
    download_model_readmes(hf_readme_download_dir, overwrite)
