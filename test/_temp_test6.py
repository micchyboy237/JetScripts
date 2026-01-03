from typing import List

import acoustid
import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

def fingerprint_audio_bytes(audio_bytes: bytes, sample_rate: int = 44100) -> str:
    """Generate Chromaprint fingerprint from raw PCM bytes (mono, int16 assumed)."""
    duration, fingerprint = acoustid.fingerprint(audio_bytes, sample_rate)
    return fingerprint

def compute_similarity_matrix(audio_bytes_list: List[bytes], sample_rate: int = 44100) -> np.ndarray:
    """Compute pairwise similarity scores (0.0 to 1.0, higher = more similar)."""
    fingerprints = [
        fingerprint_audio_bytes(ab, sample_rate) for ab in tqdm(audio_bytes_list, desc="Fingerprinting")
    ]
    n = len(fingerprints)
    sim_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="Comparing pairs"):
        for j in range(i, n):
            score = acoustid.compare_fingerprints(fingerprints[i], fingerprints[j])
            sim_matrix[i, j] = sim_matrix[j, i] = score
    return sim_matrix

def print_similarity_table(sim_matrix: np.ndarray, labels: List[str] | None = None) -> None:
    table = Table(title="Audio Similarity Matrix")
    if labels is None:
        labels = [f"Audio {i+1}" for i in range(len(sim_matrix))]
    for label in labels:
        table.add_column(label, justify="right")
    for i, row in enumerate(sim_matrix):
        table.add_row(*[f"{val:.3f}" for val in row], title=labels[i])
    console.print(table)

# Example usage (replace with real bytes)
# audio_list: List[bytes] = [bytes1, bytes2, bytes3]
# sim = compute_similarity_matrix(audio_list)
# print_similarity_table(sim, ["Clip A", "Clip B", "Clip C"])