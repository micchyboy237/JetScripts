from typing import List
from jet.audio.audio_search import AudioSegmentDatabase
from jet.audio.utils import resolve_audio_paths
from rich.console import Console
from tqdm import tqdm

console = Console()


def demo_index_and_search_files(persist_dir: str = "./my_audio_db"):
    """
    Demo: Index a directory of audio files (using the legacy file-based method)
    and perform searches with both file path and bytes queries.
    Useful for initial indexing of existing files on disk.
    """
    db = AudioSegmentDatabase(persist_dir=persist_dir)

    # Example 1: Index some audio files (run once)
    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    for file in tqdm(audio_files, desc="Indexing files"):
        db.add_segments(file)

    # Example 2: Search with a query file
    query_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments/segment_002/sound.wav"
    console.print("[bold cyan]Searching with file path query[/bold cyan]")
    results = db.search_similar(query_path, top_k=10)
    db.print_results(results)

    # Example 3: Search with raw audio bytes (e.g., from API upload)
    with open(query_path, "rb") as f:
        query_bytes = f.read()

    console.print("[bold cyan]Searching with raw bytes query[/bold cyan]")
    results_bytes = db.search_similar(query_bytes, top_k=5)
    db.print_results(results_bytes)


def demo_bytes_only_workflow(persist_dir: str = "./my_bytes_audio_db"):
    """
    Demo: Create a separate database and add/search using only raw bytes.
    Ideal for in-memory pipelines, web uploads, or when no file paths are available.
    """
    db = AudioSegmentDatabase(persist_dir=persist_dir)

    # Load some example audio as bytes (in real use: from request.files, microphone buffer, etc.)
    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    for i, path in enumerate(audio_files):
        with open(path, "rb") as f:
            audio_bytes = f.read()
        # Use a meaningful name or generate one (e.g., from upload filename)
        audio_name = f"uploaded_segment_{i+1:03d}"
        db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=None,  # whole audio as single segment
        )

    # Query with bytes from a new/unseen segment
    query_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments/segment_004/sound.wav"
    with open(query_path, "rb") as f:
        query_bytes = f.read()

    console.print("[bold cyan]Searching the bytes-only database with new audio bytes[/bold cyan]")
    results = db.search_similar(query_bytes, top_k=8)
    db.print_results(results)


def _get_sample_audio_files() -> List[str]:
    # Example 1: Index some audio files (run once)
    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments"
    audio_files = resolve_audio_paths(audio_dir, recursive=True)
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    return audio_files


if __name__ == "__main__":
    # Run one or both demos
    demo_index_and_search_files()
    demo_bytes_only_workflow()
