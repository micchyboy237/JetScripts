import io
import json
from pathlib import Path
from typing import List
from datetime import datetime

import torch
import torchaudio
from rich.console import Console
from rich.table import Table
from speechbrain.inference import EncoderClassifier
from tqdm import tqdm

import os
import shutil

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DATA_DIR = OUTPUT_DIR / "data"
RESULTS_DIR = OUTPUT_DIR / "results"
shutil.rmtree(RESULTS_DIR, ignore_errors=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

console = Console()

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(DATA_DIR / "pretrained_ecapa_tdnn")  # Auto-downloads on first run
)


def get_embedding(audio_bytes: bytes, target_sr: int = 16000, min_samples: int = 3200) -> torch.Tensor:
    """Extract normalized embedding from raw bytes (any format torchaudio supports).
    
    Handles very short or empty audio by padding with zeros to a minimum length.
    This prevents convolution padding errors in ECAPA-TDNN for ultra-short clips.
    """
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:  # Stereo to mono
        waveform = waveform.mean(0, keepdim=True)
    
    # Pad short waveforms to avoid conv padding > input size errors
    if waveform.shape[1] < min_samples:
        padding = min_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
    return embedding.squeeze()

def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two 1D embedding tensors.
    Both tensors are expected to be 1D (embedding_dim,).
    Returns a float in range [-1.0, 1.0].
    """
    # Ensure 1D tensors
    if emb1.dim() > 1:
        emb1 = emb1.squeeze()
    if emb2.dim() > 1:
        emb2 = emb2.squeeze()
    
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()


def compute_similarity_matrix(audio_bytes_list: List[bytes]) -> List[List[float]]:
    embeddings = [get_embedding(ab) for ab in tqdm(audio_bytes_list, desc="Extracting embeddings")]
    n = len(embeddings)
    sim_matrix = []
    for i in tqdm(range(n), desc="Cosine comparisons"):
        row = []
        for j in range(n):
            score = cosine_similarity(embeddings[i], embeddings[j])
            row.append(score)
        sim_matrix.append(row)
    return sim_matrix


def print_similarity_table(sim_matrix: List[List[float]], labels: List[str] | None = None) -> None:
    table = Table(title="Audio Similarity Matrix (Cosine, -1 to 1)", show_header=True, header_style="bold magenta")
    if labels is None:
        labels = [f"Audio {i+1}" for i in range(len(sim_matrix))]
    
    # Add a corner cell + column headers
    table.add_column("", style="dim")  # Empty corner for row labels
    for label in labels:
        # Truncate long labels to avoid breaking the table layout
        truncated = label[-30:] if len(label) > 30 else label
        table.add_column(truncated, justify="right")
    
    # Add rows with row label in first column
    for i, row in enumerate(sim_matrix):
        truncated_row_label = labels[i][-30:] if len(labels[i]) > 30 else labels[i]
        table.add_row(truncated_row_label, *[f"{val:+.3f}" for val in row])
    
    console.print(table)


def find_similar_groups(
    sim_matrix: List[List[float]],
    labels: List[str],
    threshold: float = 0.95
) -> List[List[str]]:
    """
    Find groups of audio files that are mutually similar above the given threshold.
    
    Returns a list of groups (each group is a list of labels). 
    Groups are sorted by size descending. Singleton groups are excluded.
    """
    n = len(sim_matrix)
    visited = [False] * n
    groups: List[List[str]] = []
    
    for i in range(n):
        if visited[i]:
            continue
        
        # Start a new group with i
        group_indices = [i]
        visited[i] = True
        
        # Find all j where sim(i,j) >= threshold AND sim(j,i) >= threshold
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i][j] >= threshold and sim_matrix[j][i] >= threshold:
                group_indices.append(j)
                visited[j] = True
        
        if len(group_indices) > 1:
            group_labels = [labels[idx] for idx in group_indices]
            groups.append(sorted(group_labels))
    
    # Sort groups by size descending
    groups.sort(key=len, reverse=True)
    return groups


def print_similar_groups(groups: List[List[str]]) -> None:
    """Pretty-print the detected similar/duplicate audio groups."""
    if not groups:
        console.print("[green]No similar audio groups found (all unique).[/green]")
        return
    
    console.print(f"[bold green]Found {len(groups)} group(s) of similar audio files:[/bold green]")
    for idx, group in enumerate(groups, 1):
        table = Table(title=f"Group {idx} ({len(group)} files)")
        table.add_column("File Label", style="cyan")
        for label in group:
            table.add_row(label)
        console.print(table)
        console.print()  # Empty line between groups


def save_demo_results(
    demo_name: str,
    sim_matrix: List[List[float]],
    labels: List[str],
    groups: List[List[str]],
    threshold: float | None = None
) -> None:
    """
    Save similarity matrix, labels, groups, and metadata for a demo run.
    Creates a subdirectory under RESULTS_DIR named demo_<timestamp>_<demo_name>.
    Saves everything in a single human-readable JSON file.
    """
    subdir_name = f"demo_{demo_name}"
    demo_dir = RESULTS_DIR / subdir_name
    demo_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "demo_name": demo_name,
        "timestamp": datetime.now().isoformat(),
        "threshold_used": threshold,
        "labels": labels,
        "similarity_matrix": sim_matrix,
        "similar_groups": groups,
    }

    json_path = demo_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)

    console.print(f"[green]Saved demo results to:[/green] {json_path}")


def demo_exact_duplicates() -> None:
    console.print("[bold blue]Demo 1: Exact and near-duplicate audio segments[/bold blue]")
    
    # Default fallback values
    default_sr = 16000
    default_waveform = torch.zeros(1, 32000)  # ~2 seconds of silence
    
    sample_bytes: bytes
    sr: int
    waveform: torch.Tensor
    
    try:
        # Attempt to download and use a real speech sample
        asset_path = torchaudio.utils.download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
        )
        waveform, sr = torchaudio.load(asset_path)
        waveform = waveform[:, :32000]  # Trim to ~2 seconds
    except Exception:
        # Fallback to silence if download/network fails
        waveform = default_waveform
        sr = default_sr
    
    # Save base clip to bytes
    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sr, format="wav")
    sample_bytes = buf.getvalue()
    
    # Create near-duplicate with very small noise
    waveform_noisy = waveform + torch.randn_like(waveform) * 0.001
    buf_noisy = io.BytesIO()
    torchaudio.save(buf_noisy, waveform_noisy, sr, format="wav")
    noisy_bytes = buf_noisy.getvalue()
    
    # Different content (longer silence)
    waveform_diff = torch.zeros(1, 48000)  # ~3 seconds silence
    buf_diff = io.BytesIO()
    torchaudio.save(buf_diff, waveform_diff, default_sr, format="wav")
    diff_bytes = buf_diff.getvalue()
    
    demo_bytes = [
        sample_bytes,      # 0
        sample_bytes,      # 1 - exact duplicate of 0
        sample_bytes,      # 2
        noisy_bytes,       # 3 - very close to 2
        diff_bytes         # 4 - different
    ]
    
    demo_labels = [
        "segment_exact_1",
        "segment_exact_2_duplicate",
        "segment_near_1",
        "segment_near_2_noisy",
        "segment_different"
    ]
    
    sim_matrix = compute_similarity_matrix(demo_bytes)
    print_similarity_table(sim_matrix, demo_labels)
    
    groups = find_similar_groups(sim_matrix, demo_labels, threshold=0.95)
    print_similar_groups(groups)
    
    save_demo_results(
        demo_name="exact_duplicates",
        sim_matrix=sim_matrix,
        labels=demo_labels,
        groups=groups,
        threshold=0.95
    )


def demo_random_short_clips() -> None:
    console.print("[bold blue]Demo 2: Random unrelated short audio clips[/bold blue]")
    
    demo_bytes_list: List[bytes] = []
    demo_labels_list: List[str] = []
    
    for i in range(6):
        # Random length between 0.1 and 0.5 seconds
        length = int(16000 * (0.1 + 0.4 * i / 5))
        waveform = torch.randn(1, length) * 0.05  # low amplitude noise
        buf = io.BytesIO()
        torchaudio.save(buf, waveform, 16000, format="wav")
        demo_bytes_list.append(buf.getvalue())
        demo_labels_list.append(f"random_noise_{i+1:02d}")
    
    sim_matrix = compute_similarity_matrix(demo_bytes_list)
    print_similarity_table(sim_matrix, demo_labels_list)
    
    groups = find_similar_groups(sim_matrix, demo_labels_list, threshold=0.90)
    print_similar_groups(groups)
    
    save_demo_results(
        demo_name="random_short_clips",
        sim_matrix=sim_matrix,
        labels=demo_labels_list,
        groups=groups,
        threshold=0.90
    )


def demo_local_files():
    from jet.audio.utils import resolve_audio_paths, load_audio_files_to_bytes, extract_audio_segment
    from pathlib import Path
    import torchaudio
    import numpy as np
    import tempfile
    import os

    console.print("[bold blue]Demo: Local files + partial query segment comparison[/bold blue]")

    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments"
    audio_files = resolve_audio_paths(audio_dir, recursive=True)
    audio_files = audio_files[:3]  # Use first 3 full segments for comparison

    # Define the source file and extract a short partial segment (first 0.6 seconds)
    full_query_path = Path(audio_dir) / "segment_0002" / "sound.wav"
    partial_waveform, sr = extract_audio_segment(str(full_query_path), start=0.0, end=0.6)

    # Convert to mono float32 tensor if needed
    partial_audio = np.asarray(partial_waveform, dtype=np.float32)
    if partial_audio.ndim == 2:
        partial_audio = partial_audio.mean(axis=0)  # Average channels for mono
    elif partial_audio.ndim > 2:
        raise ValueError("Unexpected audio array dimensions")

    audio_tensor = torch.from_numpy(partial_audio).unsqueeze(0)  # [1, samples]

    # Save to temporary file so we can reuse load_audio_files_to_bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        torchaudio.save(tmp_file.name, audio_tensor, sr)
        temp_partial_path = tmp_file.name

    # Load all audio bytes: full segments + the partial query
    all_paths = audio_files + [temp_partial_path]
    audio_bytes_list = load_audio_files_to_bytes(all_paths)

    # Clean up temp file after loading
    os.unlink(temp_partial_path)

    # Labels: keep original naming for full files, special label for partial
    labels = [
        str(Path(p).parent.name) + '_' + Path(p).name.replace('.', '_')
        for p in audio_files
    ]
    labels.append("partial_query_0.0-0.6s_from_segment_0002")

    sim_matrix = compute_similarity_matrix(audio_bytes_list)
    print_similarity_table(sim_matrix, labels)

    groups = find_similar_groups(sim_matrix, labels, threshold=0.95)
    print_similar_groups(groups)

    save_demo_results(
        demo_name="local_files_with_partial_query",
        sim_matrix=sim_matrix,
        labels=labels,
        groups=groups,
        threshold=0.95
    )

def demo_subsegment_search():
    from jet.audio.utils import resolve_audio_paths, extract_audio_segment
    import numpy as np
    from pathlib import Path
    import tempfile
    import os

    console.print("[bold blue]Demo: Sub-segment search - query from 0.2-0.8 s inside segment_0002[/bold blue]")

    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments"
    candidate_paths = resolve_audio_paths(audio_dir, recursive=True)[:10]

    # Query: now a true inner partial - 0.2 s to 0.8 s of segment_0002 (duration = 0.6 s)
    query_source_path = Path(audio_dir) / "segment_0002" / "sound.wav"
    if not query_source_path.exists():
        console.print(f"[red]Query source file not found:[/red] {query_source_path}")
        return

    query_waveform, sr = extract_audio_segment(str(query_source_path), start=0.2, end=0.8)

    query_audio = np.asarray(query_waveform, dtype=np.float32)
    if query_audio.ndim == 2:
        query_audio = query_audio.mean(axis=0)
    query_tensor = torch.from_numpy(query_audio).unsqueeze(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, query_tensor, sr)
        query_path = tmp.name

    query_bytes = Path(query_path).read_bytes()
    os.unlink(query_path)

    query_emb = get_embedding(query_bytes)
    query_duration = len(query_audio) / sr  # ~0.6 s

    # Adaptive window: use query duration as window size
    window_sec: float = query_duration
    hop_sec: float = max(0.1, window_sec / 4)  # minimum 0.1 s hop for short queries
    window_samples = int(sr * window_sec)
    hop_samples = int(sr * hop_sec)

    console.print(f"[cyan]Query duration:[/cyan] {query_duration:.3f}s (0.2-0.8 s of segment_0002)")
    console.print(f"[cyan]Window size:[/cyan] {window_sec:.3f}s, hop {hop_sec:.3f}s")

    results = []
    best_matching_window_bytes: bytes | None = None
    best_score = -1.0
    best_candidate_label = ""
    best_window_start = 0.0

    for cand_path in tqdm(candidate_paths, desc="Processing candidate segments"):
        cand_path = Path(cand_path)
        if not cand_path.exists():
            continue

        waveform, file_sr = torchaudio.load(cand_path)
        if file_sr != sr:
            waveform = torchaudio.functional.resample(waveform, file_sr, sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.squeeze(0)

        total_samples = waveform.shape[0]
        duration_sec = total_samples / sr
        label = str(cand_path.parent.name) + '_' + cand_path.name.replace('.', '_')

        # Handle short segments (pad if needed)
        if total_samples < window_samples:
            padding = window_samples - total_samples
            window_padded = torch.nn.functional.pad(waveform.unsqueeze(0), (0, padding)).squeeze(0)
            usable_windows = [window_padded]
            window_starts_sec = [0.0]
        else:
            usable_windows = []
            window_starts_sec = []
            start_sample = 0
            while start_sample <= total_samples - window_samples:
                end_sample = start_sample + window_samples
                usable_windows.append(waveform[start_sample:end_sample])
                window_starts_sec.append(start_sample / sr)
                start_sample += hop_samples

            # Add last overlapping window if meaningful
            if start_sample < total_samples and total_samples - start_sample >= int(sr * 0.3):
                last_window = waveform[-window_samples:]
                usable_windows.append(last_window)
                window_starts_sec.append((total_samples - window_samples) / sr)

        local_max_score = -1.0
        local_best_start = 0.0
        local_best_window_bytes = None

        for win_tensor, start_sec in zip(usable_windows, window_starts_sec):
            buf = io.BytesIO()
            torchaudio.save(buf, win_tensor.unsqueeze(0), sr, format="wav")
            win_bytes = buf.getvalue()

            win_emb = get_embedding(win_bytes)
            score = cosine_similarity(query_emb, win_emb)

            if score > local_max_score:
                local_max_score = score
                local_best_start = start_sec
                local_best_window_bytes = win_bytes

            if score > best_score:
                best_score = score
                best_matching_window_bytes = win_bytes
                best_candidate_label = label
                best_window_start = start_sec

        results.append({
            "candidate": label,
            "duration_sec": round(duration_sec, 3),
            "best_window_start_sec": round(local_best_start, 3),
            "best_window_end_sec": round(local_best_start + window_sec, 3),
            "max_cosine_score": round(local_max_score, 4)
        })

    if not results:
        console.print("[red]No candidates processed.[/red]")
        return

    results.sort(key=lambda x: x["max_cosine_score"], reverse=True)

    table = Table(title="Top matching sub-segments (query = 0.2-0.8 s inside segment_0002)")
    table.add_column("Rank", justify="right")
    table.add_column("Candidate File")
    table.add_column("Duration (s)")
    table.add_column("Best Window (sec)")
    table.add_column("Cosine Score", justify="right")
    for rank, res in enumerate(results, 1):
        window_str = f"{res['best_window_start_sec']} - {res['best_window_end_sec']}"
        table.add_row(
            str(rank),
            res['candidate'],
            f"{res['duration_sec']:.2f}",
            window_str,
            f"{res['max_cosine_score']:.4f}"
        )
    console.print(table)

    # Direct comparison of query vs absolute best-matching sub-segment
    console.print("\n[bold magenta]Direct comparison: Query (0.2-0.8 s) vs Best-matching sub-segment[/bold magenta]")

    if best_matching_window_bytes is None:
        console.print("[red]No matching window found.[/red]")
    else:
        best_window_emb = get_embedding(best_matching_window_bytes)
        final_score = cosine_similarity(query_emb, best_window_emb)

        top_window_str = f"{best_window_start:.3f} - {best_window_start + window_sec:.3f} s"

        console.print("Query: 0.200 - 0.800 s from segment_0002")
        console.print(f"↔ Best sub-segment: {best_candidate_label} @ {top_window_str}")
        console.print(f"[bold green]Cosine similarity:[/bold green] {final_score:.6f}")

        if abs(final_score - 1.0) < 1e-4:
            console.print("[bold green]✓ Perfect match - the window exactly corresponds to the query interval[/bold green]")
        elif final_score > 0.85:
            console.print("[bold cyan]✓ Very strong match (near-identical content)[/bold cyan]")
        else:
            console.print("[yellow]Moderate match - short duration limits embedding reliability[/yellow]")

    # In demo_subsegment_search(), before save_demo_results()
    results_data = {
        "query_interval": "0.200 - 0.800 s from segment_0002",
        "top_matches": [
            {
                "rank": r,
                "candidate": res["candidate"],
                "duration_sec": res["duration_sec"],
                "best_window": f"{res['best_window_start_sec']} - {res['best_window_end_sec']}",
                "score": res["max_cosine_score"]
            }
            for r, res in enumerate(results, 1)
        ],
        "direct_comparison": {
            "query": "0.200 - 0.800 s",
            "best_subsegment": f"{best_candidate_label} @ {best_window_start:.3f} - {best_window_start + window_sec:.3f} s",
            "cosine_score": round(best_score, 6),
            "verdict": "Very strong match" if best_score > 0.85 else "Moderate"
        }
    }

    # Then update save_demo_results() call:
    save_demo_results(
        demo_name="subsegment_search_inner_partial_0.2-0.8s",
        sim_matrix=results_data,  # Or add as new key
        labels=[r["candidate"] for r in results],
        groups=[],
        threshold=None
    )


import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import List

class SpeechbrainEmbeddingFunction(EmbeddingFunction[List[bytes]]):
    """ChromaDB embedding function that accepts list of raw audio bytes and returns ECAPA-TDNN embeddings."""
    
    def __init__(self):
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(DATA_DIR / "pretrained_ecapa_tdnn")
        )
    
    def __call__(self, input: List[bytes]) -> List[List[float]]:
        embeddings = []
        for audio_bytes in tqdm(input, desc="Extracting embeddings for Chroma"):
            emb = get_embedding(audio_bytes)  # Reuse existing robust get_embedding
            embeddings.append(emb.tolist())
        return embeddings


def demo_chroma_db() -> None:
    console.print("[bold blue]Demo: ChromaDB persistent vector store for audio embeddings[/bold blue]")
    
    # Sample audio data (reuse logic from demo_exact_duplicates for reproducibility)
    default_sr = 16000
    default_waveform = torch.zeros(1, 32000)  # ~2s silence fallback
    
    try:
        asset_path = torchaudio.utils.download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
        )
        waveform, sr = torchaudio.load(asset_path)
        waveform = waveform[:, :32000]
    except Exception:
        waveform = default_waveform
        sr = default_sr
    
    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sr, format="wav")
    base_bytes = buf.getvalue()
    
    # Create variants
    exact_dup = base_bytes
    
    noisy = waveform + torch.randn_like(waveform) * 0.001
    buf_noisy = io.BytesIO()
    torchaudio.save(buf_noisy, noisy, sr, format="wav")
    noisy_bytes = buf_noisy.getvalue()
    
    different = torch.zeros(1, 48000)
    buf_diff = io.BytesIO()
    torchaudio.save(buf_diff, different, sr, format="wav")
    diff_bytes = buf_diff.getvalue()
    
    # Collection data
    audio_bytes_list = [
        base_bytes,      # 0
        exact_dup,       # 1
        base_bytes,      # 2
        noisy_bytes,     # 3
        diff_bytes       # 4
    ]
    
    ids = [f"clip_{i:03d}" for i in range(len(audio_bytes_list))]
    metadatas = [
        {"label": "original_speech", "type": "speech"},
        {"label": "exact_duplicate", "type": "duplicate"},
        {"label": "original_speech_again", "type": "speech"},
        {"label": "noisy_version", "type": "near_duplicate"},
        {"label": "silence_different", "type": "different"}
    ]
    
    # Initialize Chroma client (persistent)
    persist_dir = DATA_DIR / "chroma_persistent"
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    embedding_fn = SpeechbrainEmbeddingFunction()
    
    collection_name = "audio_embeddings_demo"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # Use cosine distance (critical for speaker embeddings)
    )
    
    console.print(f"[cyan]Adding {len(audio_bytes_list)} audio clips to collection '{collection_name}'...[/cyan]")
    
    # Optional: Clear existing for demo reproducibility (remove if you want true persistence across runs)
    if collection.count() > 0:
        console.print("[yellow]Existing documents found – deleting for clean demo[/yellow]")
        collection.delete(ids=ids)  # Or collection.delete() to wipe all
    
    collection.add(
        ids=ids,
        embeddings=embedding_fn(audio_bytes_list),
        metadatas=metadatas,
        documents=ids
    )
    
    console.print(f"[green]Added to persistent DB at:[/green] {persist_dir}")
    
    # Query with the noisy version
    query_bytes = noisy_bytes
    # query_id = "query_noisy_clip"  # <--- Removed, no longer needed
    
    console.print("[cyan]Querying collection with noisy clip (should match originals + exact dups highly)...[/cyan]")
    results = collection.query(
        query_embeddings=embedding_fn([query_bytes]),
        n_results=5,
        include=["metadatas", "documents", "distances"]
    )
    
    # Print results table
    table = Table(title="ChromaDB Query Results (noisy clip query)")
    table.add_column("Rank", justify="right")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Type")
    table.add_column("Distance", justify="right")
    table.add_column("Cosine Similarity", justify="right")
    
    for rank, (id_, meta, dist) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["distances"][0]), 1):
        similarity = 1 - dist  # Correct conversion for cosine distance (range 0–2 → similarity 1–-1)
        table.add_row(
            str(rank),
            id_,
            meta["label"],
            meta["type"],
            f"{dist:.4f}",
            f"{similarity:+.4f}"
        )
    
    console.print(table)
    
    # Highlight high-confidence matches
    high_match_ids = [
        id_ for id_, dist in zip(results["ids"][0], results["distances"][0])
        if (1 - dist) > 0.95
    ]
    console.print(f"[bold green]High-confidence matches (>0.95 cosine):[/bold green] {high_match_ids}")
    
    # Save demo info
    demo_results = {
        "demo_name": "chromadb_persistent_demo",
        "timestamp": datetime.now().isoformat(),
        "collection_name": collection_name,
        "persist_directory": str(persist_dir),
        "total_documents": collection.count(),
        "query_description": "noisy version of original speech clip",
        "top_matches": [
            {
                "id": id_,
                "label": meta["label"],
                "cosine_similarity": round(1 - dist, 6)
            }
            for id_, meta, dist in zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
        ]
    }
    
    demo_dir = RESULTS_DIR / "demo_chromadb"
    demo_dir.mkdir(parents=True, exist_ok=True)
    json_path = demo_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2)
    
    console.print(f"[green]ChromaDB demo results saved to:[/green] {json_path}")
    console.print(f"[green]Persistent database available at:[/green] {persist_dir}")
    console.print("[yellow]Note: You can reload this DB later with PersistentClient(path=...) and get_collection('audio_embeddings_demo')[/yellow]")


if __name__ == "__main__":
    demo_exact_duplicates()
    console.print("\n" + "="*80 + "\n")
    demo_random_short_clips()
    console.print("\n" + "="*80 + "\n")
    demo_local_files()
    console.print("\n" + "="*80 + "\n")
    demo_subsegment_search()
    console.print("\n" + "="*80 + "\n")
    demo_chroma_db()