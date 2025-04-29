import mlx.core as mx
import mlx.nn as nn
from safetensors.torch import load_file
import numpy as np
import json
from pathlib import Path


def convert_safetensors_to_npz(model_path: str, output_path: str):
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Load config.json
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    # Load safetensors weights
    safetensors_file = model_path / "model.safetensors"
    if not safetensors_file.exists():
        raise FileNotFoundError(f"Could not find {safetensors_file}")

    weights = load_file(safetensors_file)

    # If model.safetensors.index.json exists, handle sharded weights
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
        # Load additional safetensors files if sharded
        for safetensors_shard in index.get("weight_map", {}).values():
            shard_path = model_path / safetensors_shard
            if shard_path != safetensors_file:
                weights.update(load_file(shard_path))

    # Convert weights to MLX format (NumPy arrays)
    np_weights = {}
    for key, tensor in weights.items():
        # Convert PyTorch tensor to NumPy array
        np_weights[key] = tensor.numpy()

    # Save weights to .npz format
    np.savez(output_path / "weights.npz", **np_weights)
    print(f"Saved weights to {output_path / 'weights.npz'}")


if __name__ == "__main__":
    model_path = "Llama-3.2-1B-Instruct-4bit"  # Path to your model directory
    output_path = model_path  # Save weights.npz in the same directory
    convert_safetensors_to_npz(model_path, output_path)
