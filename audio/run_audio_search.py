from typing import List

import torch
import torchaudio
from jet.audio.audio_search import AudioSegmentDatabase
from jet.audio.utils import extract_audio_segment, resolve_audio_paths
import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from pathlib import Path  # Needed for Path operations

import shutil

console = Console()


def _get_demo_output_dir(sub_dir: str) -> Path:
    """
    Create and return a dedicated output directory for the calling demo function.
    Directory name is based on the demo function name for clear separation.
    Existing content is removed to ensure clean runs.
    """
    import inspect

    caller_frame = inspect.stack()[1]
    func_name = caller_frame.function
    script_base = Path(__file__).parent / "generated" / Path(__file__).stem
    results_dir = script_base / "results" / func_name / sub_dir
    shutil.rmtree(results_dir, ignore_errors=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"[bold blue]Using demo results directory: {results_dir}[/bold blue]")
    return results_dir


def _get_demo_persist_dir() -> Path:
    """
    Return a dedicated Chroma persistence directory for the calling demo.
    All databases are centralized under generated/<script>/db/<demo_name>
    The DB directory is NOT cleaned on each run — allowing reuse across runs.
    """
    import inspect

    caller_frame = inspect.stack()[1]
    func_name = caller_frame.function
    script_base = Path(__file__).parent / "generated" / Path(__file__).stem
    db_base = script_base / "db"
    demo_persist_dir = db_base / func_name
    shutil.rmtree(demo_persist_dir, ignore_errors=True)

    demo_persist_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"[bold magenta]Using Chroma DB directory (persistent): {demo_persist_dir}[/bold magenta]")
    return demo_persist_dir


def _save_search_results(
    output_dir: Path,
    demo_name: str,
    results: List[dict],
    suffix: str = "",
) -> None:
    """
    Save search results to JSON (full data) and Markdown (human-readable table).
    Uses the same table format as print_results for consistency.
    """
    import json

    suffix_part = f"_{suffix}" if suffix else ""
    json_path = output_dir / f"search_results{suffix_part}.json"
    md_path = output_dir / f"search_results{suffix_part}.md"

    # Save full results as JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Generate proper GitHub-flavored markdown table directly (no Rich capture needed)
    md_lines = []
    md_lines.append(f"# Search Results{suffix_part.replace('_', ' ')} - {demo_name}")
    md_lines.append("")
    md_lines.append("| Rank | ID | File | Time Range | Similarity |")
    md_lines.append("| ------ | ---- | ------ | ------------ | ------------ |")

    for rank, res in enumerate(results, 1):
        time_range = f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s"
        file_name = Path(res["file"]).name if res["file"] != "<bytes>" else "<bytes>"
        md_lines.append(
            f"| {rank} | {res['id']} | {file_name} | {time_range} | {res['score']:.4f} |"
        )

    md_lines.append("")
    markdown_content = "\n".join(md_lines)

    with open(md_path, "w") as f:
        f.write(markdown_content)

    # Additional: save growing prefix progression table if present
    if results and "prefix_scores" in results[0]:
        prog_md_path = output_dir / f"growing_prefix_progression{suffix_part}.md"
        prog_lines = []
        prog_lines.append("# Growing Prefix Score Progression")
        prog_lines.append("")
        prog_lines.append("| Rank | File | Prefix Duration | Score History |")
        prog_lines.append("| ------ | ------ | ------------------ | --------------- |")

        for rank, res in enumerate(results[:10], 1):
            durations = [f"{d:.1f}s" for d in res["prefix_durations_sec"]]
            scores = [f"{s:.3f}" for s in res["prefix_scores"]]
            duration_str = " → ".join(durations)
            score_str = " → ".join(scores)
            file_name = Path(res["file"]).name if res["file"] != "<bytes>" else "<bytes>"
            prog_lines.append(f"| {rank} | {file_name} | {duration_str} | {score_str} |")

        prog_lines.append("")
        prog_markdown = "\n".join(prog_lines)

        with open(prog_md_path, "w") as f:
            f.write(prog_markdown)

        console.log(f"[bold green]Saved growing prefix progression to {prog_md_path}[/bold green]")

    # ---- Additional: save localization results table if localization info is present ----
    # (copied pattern from growing prefix table, using instructions)
    # This expects that if any result in `results` has "query_start_sec" in its dict, we produce the table.
    if results and "query_start_sec" in results[0]:
        loc_lines = []
        loc_lines.append("# Localization Results")
        loc_lines.append("")
        loc_lines.append("| Rank | ID | File | DB Time | Query Time | Similarity |")
        loc_lines.append("| ------ | ---- | ------ | --------- | ------------ | ------------ |")
        for rank, res in enumerate(results, 1):
            if "query_start_sec" not in res:
                continue
            db_time = f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s"
            query_time = f"{res['query_start_sec']:.1f}s – {res['query_end_sec']:.1f}s"
            loc_lines.append(
                f"| {rank} | {res['id']} | {Path(res['file']).name} | {db_time} | {query_time} | {res['score']:.4f} |"
            )
        loc_lines.append("")
        loc_markdown = "\n".join(loc_lines)
        loc_md_path = output_dir / f"localization_results{suffix_part}.md"
        with open(loc_md_path, "w") as f:
            f.write(loc_markdown)

    console.log(f"[bold green]Saved search results to {json_path} and {md_path}[/bold green]")


def demo_index_and_search_files(query_path: str, out_dir: str = "full"):
    """
    Demo: Index a directory of audio files (using the legacy file-based method)
    and perform searches with both file path and bytes queries.
    Useful for initial indexing of existing files on disk.
    """
    console.log("[bold yellow]Starting demo: demo_index_and_search_files[/bold yellow]")
    output_dir = _get_demo_output_dir(out_dir)

    persist_dir_path = _get_demo_persist_dir()
    db = AudioSegmentDatabase(persist_dir=str(persist_dir_path))

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
            # console.print(f"[dim]Already indexed: {Path(file).name}[/dim]")
            continue

        db.add_segments(file)  # only call if needed

    # Example 2: Search with a query file
    console.print("[bold cyan]Searching with file path query[/bold cyan]")
    results = db.search_similar(query_path, top_k=10)
    console.log("[green]Printing search results for file path query[/green]")
    db.print_results(results)
    _save_search_results(output_dir, "demo_index_and_search_files", results, suffix="file_path")

    # Example 3: Search with raw audio bytes (e.g., from API upload)
    with open(query_path, "rb") as f:
        query_bytes = f.read()

    console.print("[bold cyan]Searching with raw bytes query[/bold cyan]")
    results_bytes = db.search_similar(query_bytes, top_k=5)
    console.log("[green]Printing search results for raw bytes query[/green]")
    db.print_results(results_bytes)
    _save_search_results(output_dir, "demo_index_and_search_files", results_bytes, suffix="bytes")


def demo_bytes_only_workflow(query_path: str, out_dir: str = "full"):
    """
    Demo: Create a separate database and add/search using only raw bytes.
    Ideal for in-memory pipelines, web uploads, or when no file paths are available.
    """
    console.log("[bold yellow]Starting demo: demo_bytes_only_workflow[/bold yellow]")
    output_dir = _get_demo_output_dir(out_dir)

    persist_dir_path = _get_demo_persist_dir()
    db = AudioSegmentDatabase(persist_dir=str(persist_dir_path))

    # Load some example audio as bytes (in real use: from request.files, microphone buffer, etc.)
    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    for i, path in enumerate(audio_files):
        audio_name = f"uploaded_segment_{i+1:03d}"
        expected_id = f"{audio_name}_full"

        existing = db.collection.get(ids=[expected_id], include=[])
        if existing["ids"]:
            # console.print(f"[dim]Already indexed: {audio_name} ({Path(path).name})[/dim]")
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
    _save_search_results(output_dir, "demo_bytes_only_workflow", results)


def demo_numpy_array_workflow(query_path: str, out_dir: str = "full"):
    """
    Demo: Index and search audio using real sample files loaded as NumPy arrays.
    Shows how to use np.ndarray input when you already have audio loaded in memory
    (e.g., from file, microphone, or API).
    """
    console.log("[bold yellow]Starting demo: demo_numpy_array_workflow[/bold yellow]")
    output_dir = _get_demo_output_dir(out_dir)

    persist_dir_path = _get_demo_persist_dir()
    db = AudioSegmentDatabase(persist_dir=str(persist_dir_path))

    # Reuse the same sample files as the other demos
    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Indexing {len(audio_files)} real audio files as NumPy arrays[/bold green]")

    # Index all sample audio files as numpy arrays
    for i, audio_path in enumerate(audio_files):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        waveform_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        db.add_segments(
            audio_input=waveform_np,
            audio_name=f"indexed_{i+1:03d}",
            segment_duration_sec=None,
        )

    # Query using the same query file loaded as NumPy array
    console.print("[bold cyan]Searching with real query audio loaded as NumPy array[/bold cyan]")
    query_waveform, query_sr = torchaudio.load(query_path)
    if query_waveform.shape[0] > 1:
        query_waveform = query_waveform.mean(dim=0)
    query_numpy = query_waveform.squeeze(0).cpu().numpy().astype(np.float32)

    results = db.search_similar(query_numpy, top_k=8)
    console.log("[green]Printing search results for NumPy query[/green]")
    db.print_results(results)
    _save_search_results(output_dir, "demo_numpy_array_workflow", results)


def demo_localize_in_query_workflow(query_path: str, out_dir: str = "full"):
    """
    Demo: Index sample audio files and search with localize_in_query=True.
    This mode chunks the query into overlapping 10-second windows and reports
    where in the query audio each database match was found.
    """
    console.log("[bold yellow]Starting demo: demo_localize_in_query_workflow[/bold yellow]")

    output_dir = _get_demo_output_dir(out_dir)
    persist_dir_path = _get_demo_persist_dir()
    db = AudioSegmentDatabase(persist_dir=str(persist_dir_path))

    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    for path in tqdm(audio_files, desc="Indexing files"):
        base_name = Path(path).stem
        db.add_segments(path, audio_name=base_name, segment_duration_sec=None)

    # For a strong demo, construct a synthetic long query containing known segments
    # at different offsets with silence in between
    console.print("[bold cyan]Building synthetic long query for clear localization demo[/bold cyan]")
    selected_files = audio_files[:3]  # Pick first 3 distinct segments
    silence_duration = 5.0  # seconds

    segments = []
    for i, p in enumerate(selected_files):
        waveform, sr = torchaudio.load(p)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        segments.append(waveform)
        if i < len(selected_files) - 1:
            silence = torch.zeros(1, int(sr * silence_duration))
            segments.append(silence)
    long_query = torch.cat(segments, dim=1)
    synthetic_path = output_dir / "synthetic_long_query.wav"
    torchaudio.save(synthetic_path, long_query, sr)

    console.print(f"[bold cyan]Synthetic query saved to {synthetic_path} "
                  f"({long_query.shape[1] / sr:.1f}s total)[/bold cyan]")
    console.print("[bold cyan]Searching with localization in query enabled[/bold cyan]")
    results = db.search_similar(synthetic_path, localize_in_query=True, top_k=10)

    console.log("[green]Standard results view[/green]")
    db.print_results(results)
    _save_search_results(output_dir, "demo_localize_in_query_workflow", results, suffix="standard")

    if results and "query_start_sec" in results[0]:
        console.print("\n[bold magenta]Localization details (time range in query)[/bold magenta]")
        loc_table = Table(title="Matches with Query Time Localization")
        loc_table.add_column("Rank", justify="right")
        loc_table.add_column("ID", style="cyan")
        loc_table.add_column("File")
        loc_table.add_column("DB Time", style="cyan")
        loc_table.add_column("Query Time", style="green")
        loc_table.add_column("Similarity", justify="right")
        for rank, res in enumerate(results, 1):
            db_time = f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s"
            query_time = f"{res['query_start_sec']:.1f}s – {res['query_end_sec']:.1f}s"
            loc_table.add_row(
                str(rank),
                res["id"],
                Path(res["file"]).name,
                db_time,
                query_time,
                f"{res['score']:.4f}"
            )
        console.print(loc_table)

        # Save localization table separately
        import json
        from rich.panel import Panel
        loc_json_path = output_dir / "localization_results.json"
        loc_md_path = output_dir / "localization_results.md"
        loc_data = [
            {
                "rank": rank,
                "id": res["id"],
                "file": Path(res["file"]).name,
                "db_time": f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s",
                "query_time": f"{res['query_start_sec']:.1f}s – {res['query_end_sec']:.1f}s",
                "similarity": res['score'],
            }
            for rank, res in enumerate(results, 1)
            if "query_start_sec" in res
        ]
        with open(loc_json_path, "w") as f:
            json.dump(loc_data, f, indent=4)
        with console.capture() as capture:
            console.print(Panel(loc_table, expand=False))
        raw_loc_capture = capture.get()

        # Replace with direct markdown generation (same pattern as other tables)
        loc_lines = []
        loc_lines.append("# Localization Results")
        loc_lines.append("")
        loc_lines.append("| Rank | ID | File | DB Time | Query Time | Similarity |")
        loc_lines.append("|------|----|------|---------|------------|------------|")
        for rank, res in enumerate(results, 1):
            if "query_start_sec" not in res:
                continue
            db_time = f"{res['start_sec']:.1f}s – {res['end_sec']:.1f}s"
            query_time = f"{res['query_start_sec']:.1f}s – {res['query_end_sec']:.1f}s"
            loc_lines.append(
                f"| {rank} | {res['id']} | {Path(res['file']).name} | {db_time} | {query_time} | {res['score']:.4f} |"
            )
        loc_lines.append("")
        loc_markdown = "\n".join(loc_lines)

        with open(loc_md_path, "w") as f:
            f.write(loc_markdown)
        console.log(f"[bold green]Saved localization results to {loc_json_path} and {loc_md_path}[/bold green]")
    else:
        # console.print("[dim]No localization info (query likely too short for windowing)[/dim]")
        pass


def demo_growing_short_segments_workflow(query_path: str, out_dir: str = "full"):
    """
    Demo: Use the new growing short segments mode.
    Useful when the query is very short or noisy — progressively longer prefixes
    of 0.1s chunks are tested until a confident match is found.
    """
    console.log("[bold yellow]Starting demo: demo_growing_short_segments_workflow[/bold yellow]")
    output_dir = _get_demo_output_dir(out_dir)

    persist_dir_path = _get_demo_persist_dir()
    db = AudioSegmentDatabase(persist_dir=str(persist_dir_path))

    audio_files = _get_sample_audio_files()
    console.print(f"[bold green]Indexing {len(audio_files)} full audio files[/bold green]")
    for path in tqdm(audio_files, desc="Indexing files"):
        db.add_segments(path, segment_duration_sec=None)

    # Create a short query: first ~1.5 seconds of the original query_path
    waveform, sr = torchaudio.load(query_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    short_samples = int(1.5 * sr)
    short_waveform = waveform[:, :short_samples]
    short_query_path = output_dir / "short_query_1.5s.wav"
    torchaudio.save(short_query_path, short_waveform, sr)
    console.print(f"[bold cyan]Created short query ({1.5}s): {short_query_path}[/bold cyan]")

    console.print("[bold cyan]Searching with growing short segments mode[/bold cyan]")
    results = db.search_similar(
        short_query_path,
        use_growing_short_segments=True,
        top_k=10
    )
    console.log("[green]Results using growing 0.1s prefixes (max score across prefixes)[/green]")
    db.print_results(results)
    _save_search_results(output_dir, "demo_growing_short_segments_workflow", results, suffix="growing")

    console.print("[bold cyan]For comparison: normal single-segment search on same short query[/bold cyan]")
    normal_results = db.search_similar(short_query_path, top_k=10)
    db.print_results(normal_results)
    _save_search_results(output_dir, "demo_growing_short_segments_workflow", normal_results, suffix="normal")


def _get_sample_audio_files() -> List[str]:
    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments"
    audio_files = resolve_audio_paths(audio_dir, recursive=True)
    console.print(f"[bold green]Found {len(audio_files)} audio files to index[/bold green]")
    return audio_files


if __name__ == "__main__":
    full_query_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0002/sound.wav"

    # Extract partial audio (0.2s to 0.8s)
    partial_audio, sr = extract_audio_segment(full_query_path, start=0.0, end=0.6)

    # Ensure mono float32 NumPy array, then convert to torch tensor with explicit channel dim
    partial_audio = np.asarray(partial_audio, dtype=np.float32)
    if partial_audio.ndim == 2:
        # If somehow stereo, convert to mono
        partial_audio = partial_audio.mean(axis=1)
    elif partial_audio.ndim > 2:
        raise ValueError("Unexpected audio array dimensions")
    # Now guaranteed 1D → add channel dimension
    audio_tensor = torch.from_numpy(partial_audio).unsqueeze(0)  # shape: [1, samples]

    # Full match query
    demo_index_and_search_files(full_query_path)
    demo_bytes_only_workflow(full_query_path)
    demo_numpy_array_workflow(full_query_path)
    demo_localize_in_query_workflow(full_query_path)
    demo_growing_short_segments_workflow(full_query_path)

    # Partial query
    temp_partial_query_path = "/tmp/temp_query_segment.wav"
    torchaudio.save(temp_partial_query_path, audio_tensor, sr)

    out_dir = "partial"
    demo_index_and_search_files(temp_partial_query_path, out_dir)
    demo_bytes_only_workflow(temp_partial_query_path, out_dir)
    demo_numpy_array_workflow(temp_partial_query_path, out_dir)
    demo_localize_in_query_workflow(temp_partial_query_path, out_dir)
    demo_growing_short_segments_workflow(temp_partial_query_path, out_dir)
