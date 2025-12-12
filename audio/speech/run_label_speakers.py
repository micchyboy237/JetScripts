import os
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table

from jet.audio.speech.pyannote.segment_speaker_labeler import SegmentSpeakerLabeler
from jet.audio.utils import resolve_audio_paths
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Replace with your actual Hugging Face token (required for pyannote/embedding)
    HF_TOKEN = "hf_your_token_here"  # or load from environment: os.getenv("HF_TOKEN")

    # Root directory containing subfolders with 'sound.wav' speech segments
    SEGMENTS_DIR = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps")

    # Clustering aggressiveness: lower threshold → more speakers detected
    DISTANCE_THRESHOLD = 0.7

    # ------------------------------------------------------------------
    # Run clustering
    # ------------------------------------------------------------------
    clusterer = SegmentSpeakerLabeler(
        embedding_model_name="pyannote/embedding",
        hf_token=HF_TOKEN,
        distance_threshold=DISTANCE_THRESHOLD,
        use_gpu=True,  # Will fallback to CPU if no CUDA available
    )

    segment_paths = resolve_audio_paths(SEGMENTS_DIR, recursive=True)
    results = clusterer.cluster_segments(
        segment_paths=segment_paths,  # Change if your segment files have different names
    )

    # ------------------------------------------------------------------
    # Quick summary (optional, using rich for pretty output)
    # ------------------------------------------------------------------

    console = Console()
    table = Table(title="Speaker Labels Summary")
    table.add_column("Parent Directory")
    table.add_column("Speaker Label")
    table.add_column("Min Cosine Sim.")
    table.add_column("Segment Path")

    for res in results:
        table.add_row(
            res["parent_dir"],
            str(res["speaker_label"]),
            f"{res['min_cosine_similarity']:.3f}",
            res["path"]
        )

    console.print(table)

    # Optional: save results to JSON
    save_file(results, f"{OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()