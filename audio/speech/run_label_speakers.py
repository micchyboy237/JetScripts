# JetScripts/audio/speech/run_label_speakers.py
import os
import shutil
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

from jet.audio.speech.pyannote.segment_speaker_labeler import SegmentResult, SegmentSpeakerLabeler
from jet.audio.utils import resolve_audio_paths
from jet.file.utils import save_file


BASE_OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_agglomerative_clustering(
    segments_dir: Path,
    hf_token: str | None,
    distance_threshold: float = 0.7,
) -> List[SegmentResult]:
    """Example: Run agglomerative clustering (default strategy – automatically estimates number of speakers)."""
    output_dir = BASE_OUTPUT_DIR / "agglomerative"
    output_dir.mkdir(parents=True, exist_ok=True)

    clusterer = SegmentSpeakerLabeler(
        embedding_model_name="pyannote/embedding",
        hf_token=hf_token,
        distance_threshold=distance_threshold,
        clustering_strategy="agglomerative",  # explicit for clarity
        use_gpu=True,
    )

    segment_paths = resolve_audio_paths(segments_dir, recursive=True)
    results = clusterer.cluster_segments(segment_paths=segment_paths)

    # Save results
    save_file(results, output_dir / "results.json")

    return results


def run_kmeans_clustering(
    segments_dir: Path,
    hf_token: str | None,
    n_clusters: int,
    distance_threshold: float = 0.7,  # unused for kmeans, kept for signature consistency
) -> List[SegmentResult]:
    """Example: Run K-Means clustering (requires known/estimated number of speakers)."""
    output_dir = BASE_OUTPUT_DIR / "kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)

    clusterer = SegmentSpeakerLabeler(
        embedding_model_name="pyannote/embedding",
        hf_token=hf_token,
        distance_threshold=distance_threshold,  # ignored by kmeans
        clustering_strategy="kmeans",
        n_clusters=n_clusters,
        use_gpu=True,
    )

    segment_paths = resolve_audio_paths(segments_dir, recursive=True)
    results = clusterer.cluster_segments(segment_paths=segment_paths)

    # Save results
    save_file(results, output_dir / "results.json")

    return results


def print_summary(results: List[SegmentResult], title: str) -> None:
    """Utility to display a clean rich table summary."""
    console = Console()
    table = Table(title=title)
    table.add_column("Parent Directory")
    table.add_column("Speaker Label")
    table.add_column("Min Cosine Sim.")
    table.add_column("Segment Path")

    for res in results:
        table.add_row(
            res["parent_dir"],
            str(res["speaker_label"]),
            f"{res['min_cosine_similarity']:.3f}",
            res["path"],
        )

    console.print(table)


def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    HF_TOKEN = os.getenv("HF_TOKEN")  # Set your Hugging Face token in environment

    SEGMENTS_DIR = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps"
    )

    # ------------------------------------------------------------------
    # Run both strategies
    # ------------------------------------------------------------------
    print("Running agglomerative clustering...")
    agg_results = run_agglomerative_clustering(
        segments_dir=SEGMENTS_DIR,
        hf_token=HF_TOKEN,
        distance_threshold=0.7,
    )
    print_summary(agg_results, title="Speaker Labels Summary – Agglomerative Clustering")

    print("\nRunning K-Means clustering (example with fixed 5 speakers)...")
    kmeans_results = run_kmeans_clustering(
        segments_dir=SEGMENTS_DIR,
        hf_token=HF_TOKEN,
        n_clusters=5,  # Change this based on your expected number of speakers
    )
    print_summary(kmeans_results, title="Speaker Labels Summary – K-Means Clustering (n=5)")


if __name__ == "__main__":
    main()