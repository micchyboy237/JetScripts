import os
from typing import Dict
from pathlib import Path

from sqlalchemy import exists
from safetensors.torch import load_file
import torch
from jet.models.model_types import ModelType
from jet.models.utils import get_local_repo_dir, resolve_model_key, resolve_model_value
import numpy as np
import json
from tqdm import tqdm
from jet.logger import logger


def convert_safetensors_to_npz(model_path: ModelType, output_path: str) -> None:
    """
    Convert safetensors weights to npz format for a given model.

    Args:
        model_path (ModelType): Model key or path to the model directory.
        output_path (str): Directory to save the output weights.npz file.
    """
    logger.info(f"Starting conversion of model {model_path} to NPZ format")

    # Resolve model path
    logger.info("Resolving model path")
    model_path = resolve_model_value(model_path)
    output_path = Path(output_path)

    # Get local repository directory
    logger.info(f"Checking for local repository directory for {model_path}")
    repo_dir = get_local_repo_dir(model_path)
    if not Path(repo_dir).exists():
        logger.error(f"Local directory not found for {model_path}")
        raise FileNotFoundError(
            f"Could not find local directory for {model_path}")

    # Load config.json
    logger.info("Loading config.json")
    config_path = get_local_repo_dir(repo_dir, "config.json")
    if not Path(config_path).exists():
        logger.error(f"Config file not found for {model_path}")
        raise FileNotFoundError(f"Could not find config.json for {model_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info(f"Loaded config from {config_path}")

    # Load safetensors weights
    logger.info("Loading primary safetensors file")
    safetensors_file = get_local_repo_dir(repo_dir, "model.safetensors")
    if not Path(safetensors_file).exists():
        logger.error(f"Safetensors file not found for {model_path}")
        raise FileNotFoundError(
            f"Could not find model.safetensors for {model_path}")

    weights = load_file(safetensors_file)
    processed_files = {str(safetensors_file)}
    logger.info(f"Loaded primary safetensors file: {safetensors_file}")

    # Handle sharded weights if index file exists
    logger.info("Checking for sharded weights index")
    index_file = get_local_repo_dir(repo_dir, "model.safetensors.index.json")
    if Path(index_file).exists():
        with open(index_file, "r") as f:
            index = json.load(f)
        shard_files = set(index.get("weight_map", {}).values())
        logger.info(f"Found {len(shard_files)} shard files in index")

        # Load shards with progress bar
        logger.info("Loading weight shards")
        for safetensors_shard in tqdm(shard_files, desc="Loading weight shards"):
            shard_path = get_local_repo_dir(repo_dir, safetensors_shard)
            if Path(shard_path).exists() and str(shard_path) not in processed_files:
                weights.update(load_file(str(shard_path)))
                processed_files.add(str(shard_path))
                logger.debug(f"Loaded shard: {shard_path}")
            else:
                logger.warning(
                    f"Shard {shard_path} not found or already processed")

    # Convert weights to NumPy format
    logger.info("Converting weights to NumPy format")
    np_weights: Dict[str, np.ndarray] = {}
    for key, tensor in tqdm(weights.items(), desc="Converting weights"):
        # Convert bfloat16 to float32 if necessary
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        np_weights[key] = tensor.numpy()
    logger.info(f"Converted {len(np_weights)} weight tensors to NumPy format")

    # Save weights to .npz format
    logger.info("Saving weights to NPZ file")
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path / "weights.npz"
    np.savez(output_file, **np_weights)
    logger.info(f"Saved weights to {output_file}")


if __name__ == "__main__":
    # Path to your model directory
    model_path: ModelType = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
    output_path = f"{os.path.dirname(__file__)}/models/{resolve_model_key(model_path)}"
    convert_safetensors_to_npz(model_path, output_path)
