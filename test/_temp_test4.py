import os
from typing import Optional
import warnings
import torch
import numpy as np
from pyannote.audio import Model, Inference

# Silence the known harmless Lightning checkpoint warning
warnings.filterwarnings(
    "ignore",
    message="Found keys that are not in the model state dict but in the checkpoint",
)

def extract_embedding(
    audio_path: str,
    model_name: str = "pyannote/embedding",
    hf_token: Optional[str] = None,
) -> torch.Tensor:
    """
    Extract a speaker embedding from an audio file (whole-file inference).

    Returns
    -------
    torch.Tensor
        1-D tensor of shape (512,) on the same device as the model (CPU/GPU).
    """
    # ------------------------------------------------------------------ #
    # Token handling
    # ------------------------------------------------------------------ #
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable or explicit token required.")

    # ------------------------------------------------------------------ #
    # Load model – strict=False removes the loss_func.W warning
    # ------------------------------------------------------------------ #
    model = Model.from_pretrained(model_name, use_auth_token=hf_token, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ------------------------------------------------------------------ #
    # Inference – returns numpy.ndarray with shape (512,)
    # ------------------------------------------------------------------ #
    inference = Inference(model, window="whole", device=device)
    embedding_np: np.ndarray = inference(audio_path)          # ← NumPy array

    # Convert to torch tensor (keeps it on CPU/GPU as needed)
    embedding = torch.from_numpy(embedding_np).to(device)

    return embedding  # shape: (512,)


# ---------------------------------------------------------------------- #
# Example usage
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav"

    embedding = extract_embedding(audio_path)

    print(f"Embedding shape : {embedding.shape}")        # (512,)
    print(f"Embedding dtype : {embedding.dtype}")
    print(f"Norm           : {torch.norm(embedding):.4f}")
    print(f"First 5 values  : {embedding[:5].tolist()}")