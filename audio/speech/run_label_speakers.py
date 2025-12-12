from pathlib import Path
from rich.console import Console
from rich.table import Table

from jet.audio.speech.pyannote.segment_speaker_labeler import SegmentSpeakerLabeler

def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Replace with your actual Hugging Face token (required for pyannote/embedding)
    HF_TOKEN = "hf_your_token_here"  # or load from environment: os.getenv("HF_TOKEN")

    # Root directory containing subfolders with 'sound.wav' speech segments
    SEGMENTS_DIR = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps")

    # Where to save the JSON results
    OUTPUT_JSON = Path("./output") / "speaker_clustering_results.json"

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

    results = clusterer.cluster_segments(
        segments_dir=SEGMENTS_DIR,
        file_pattern="**/sound.wav",  # Change if your segment files have different names
    )

    # Optional: save results to JSON
    clusterer.save_results(results, OUTPUT_JSON)

    # ------------------------------------------------------------------
    # Quick summary (optional, using rich for pretty output)
    # ------------------------------------------------------------------

    console = Console()
    table = Table(title="Speaker Labels Summary")
    table.add_column("Parent Directory")
    table.add_column("Speaker Label")
    table.add_column("Segment Path")

    for res in results:
        table.add_row(res["parent_dir"], str(res["speaker_label"]), res["path"])

    console.print(table)

if __name__ == "__main__":
    main()