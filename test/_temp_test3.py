import torch
from typing import Tuple
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference

def verify_speakers(audio1: str, audio2: str, threshold: float = 0.6, hf_token: Optional[str] = None) -> Tuple[bool, float]:
    """Verify if two audio samples are from the same speaker via cosine distance."""
    model = Model.from_pretrained("pyannote/embedding", token=hf_token)
    model.eval()
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    
    inference = Inference(model, window="whole")
    emb1: torch.Tensor = inference(audio1)
    emb2: torch.Tensor = inference(audio2)
    
    # Compute cosine distance (lower = more similar)
    distance: float = cdist(emb1.detach().numpy(), emb2.detach().numpy(), metric="cosine")[0, 0]
    is_same: bool = distance < threshold  # Tune threshold based on EER (~0.3-0.6 typical)
    
    return is_same, distance

# Usage
if __name__ == "__main__":
    same_speaker, dist = verify_speakers("speaker1.wav", "speaker1_ref.wav", hf_token="your_hf_token_here")
    print(f"Same speaker? {same_speaker}, Distance: {dist:.4f}")