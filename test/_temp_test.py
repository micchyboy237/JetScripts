from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    force_download=True,
    local_dir="/Users/jethroestrada/.cache/huggingface/hub/models--mlx-community--Qwen2.5-VL-7B-Instruct-4bit"
)
