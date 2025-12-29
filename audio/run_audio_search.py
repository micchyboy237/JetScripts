from typing import List
from jet.audio.audio_search import AudioSegmentDatabase
from jet.audio.utils import resolve_audio_paths
from rich.console import Console
from tqdm import tqdm
from pathlib import Path  # Needed for Path operations

console = Console()


def demo_index_and_search_files(query_path: str, persist_dir: str = "./my_audio_db"):
    """
    Demo: Index a directory of audio files (using the legacy file-based method)
    and perform searches with both file path and bytes queries.
    Useful for initial indexing of existing files on disk.
    """
    console.log("[bold yellow]Starting demo: demo_index_and_search_files[/bold yellow]")

    db = AudioSegmentDatabase(persist_dir=persist_dir)

    # Example 1: Index some audio files (run once)
    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    for file in tqdm(audio_files, desc="Indexing files"):
        base_name = Path(file).stem
        # Query by metadata instead (file path + full segment start/end)
        existing = db.collection.get(
            where={
                "$and": [
                    {"file": str(Path(file).resolve())},
                    {"start_sec": 0.0}
                ]
            },
            include=["metadatas"]
        )
        if existing["ids"]:
            console.print(f"[dim]Already indexed: {Path(file).name}[/dim]")
            continue

        db.add_segments(file)  # only call if needed

    # Example 2: Search with a query file
    console.print("[bold cyan]Searching with file path query[/bold cyan]")
    results = db.search_similar(query_path, top_k=10)
    console.log("[green]Printing search results for file path query[/green]")
    db.print_results(results)

    # Example 3: Search with raw audio bytes (e.g., from API upload)
    with open(query_path, "rb") as f:
        query_bytes = f.read()

    console.print("[bold cyan]Searching with raw bytes query[/bold cyan]")
    results_bytes = db.search_similar(query_bytes, top_k=5)
    console.log("[green]Printing search results for raw bytes query[/green]")
    db.print_results(results_bytes)


def demo_bytes_only_workflow(query_path: str, persist_dir: str = "./my_bytes_audio_db"):
    """
    Demo: Create a separate database and add/search using only raw bytes.
    Ideal for in-memory pipelines, web uploads, or when no file paths are available.
    """
    console.log("[bold yellow]Starting demo: demo_bytes_only_workflow[/bold yellow]")

    db = AudioSegmentDatabase(persist_dir=persist_dir)

    # Load some example audio as bytes (in real use: from request.files, microphone buffer, etc.)
    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    for i, path in enumerate(audio_files):
        audio_name = f"uploaded_segment_{i+1:03d}"
        expected_id = f"{audio_name}_full"

        existing = db.collection.get(ids=[expected_id], include=[])
        if existing["ids"]:
            console.print(f"[dim]Already indexed: {audio_name} ({Path(path).name})[/dim]")
            continue

        with open(path, "rb") as f:
            audio_bytes = f.read()

        db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=None,
        )

    # Query with bytes from a new/unseen segment
    with open(query_path, "rb") as f:
        query_bytes = f.read()

    console.print("[bold cyan]Searching the bytes-only database with new audio bytes[/bold cyan]")
    results = db.search_similar(query_bytes, top_k=8)
    console.log("[green]Printing search results for bytes-only workflow query[/green]")
    db.print_results(results)


def _get_sample_audio_files() -> List[str]:
    # Example 1: Index some audio files (run once)
    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments"
    audio_files = resolve_audio_paths(audio_dir, recursive=True)
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    return audio_files


if __name__ == "__main__":
    query_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments/segment_004/sound.wav"

    # Run one or both demos
    demo_index_and_search_files(query_path)
    demo_bytes_only_workflow(query_path)
