import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
from jet.audio.audio_search3 import (
    compute_similarity_matrix,
    cosine_similarity,
    find_similar_groups,
    get_embedding,
    print_similar_groups,
    print_similarity_table,
)
from rich.console import Console
from rich.table import Table
from speechbrain.inference import EncoderClassifier
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DATA_DIR = OUTPUT_DIR / "data"
RESULTS_DIR = OUTPUT_DIR / "results"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
# shutil.rmtree(RESULTS_DIR, ignore_errors=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

QUERY_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/sub_audio/start_5s_recording_1_speaker.wav"
AUDIO_DIR = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav",
]

console = Console()


def save_demo_results(
    demo_name: str,
    sim_matrix: list[list[float]],
    labels: list[str],
    groups: list[list[str]],
    threshold: float | None = None,
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
    console.print(
        "[bold blue]Demo 1: Exact and near-duplicate audio segments[/bold blue]"
    )

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
        sample_bytes,  # 0
        sample_bytes,  # 1 - exact duplicate of 0
        sample_bytes,  # 2
        noisy_bytes,  # 3 - very close to 2
        diff_bytes,  # 4 - different
    ]

    demo_labels = [
        "segment_exact_1",
        "segment_exact_2_duplicate",
        "segment_near_1",
        "segment_near_2_noisy",
        "segment_different",
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
        threshold=0.95,
    )


def demo_random_short_clips() -> None:
    console.print("[bold blue]Demo 2: Random unrelated short audio clips[/bold blue]")

    demo_bytes_list: list[bytes] = []
    demo_labels_list: list[str] = []

    for i in range(6):
        # Random length between 0.1 and 0.5 seconds
        length = int(16000 * (0.1 + 0.4 * i / 5))
        waveform = torch.randn(1, length) * 0.05  # low amplitude noise
        buf = io.BytesIO()
        torchaudio.save(buf, waveform, 16000, format="wav")
        demo_bytes_list.append(buf.getvalue())
        demo_labels_list.append(f"random_noise_{i + 1:02d}")

    sim_matrix = compute_similarity_matrix(demo_bytes_list)
    print_similarity_table(sim_matrix, demo_labels_list)

    groups = find_similar_groups(sim_matrix, demo_labels_list, threshold=0.90)
    print_similar_groups(groups)

    save_demo_results(
        demo_name="random_short_clips",
        sim_matrix=sim_matrix,
        labels=demo_labels_list,
        groups=groups,
        threshold=0.90,
    )


def demo_local_files():
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    import torchaudio
    from jet.audio.utils import (
        extract_audio_segment,
        load_audio_files_to_bytes,
    )

    console.print(
        "[bold blue]Demo: Local files + partial query segment comparison[/bold blue]"
    )

    candidate_paths = AUDIO_DIR
    query_path = Path(QUERY_PATH)

    partial_waveform, sr = extract_audio_segment(str(query_path), start=0.0, end=0.6)

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
    all_paths = candidate_paths + [temp_partial_path]
    audio_bytes_list = load_audio_files_to_bytes(all_paths)

    # Clean up temp file after loading
    os.unlink(temp_partial_path)

    # Labels: keep original naming for full files, special label for partial
    labels = [
        str(Path(p).parent.name) + "_" + Path(p).name.replace(".", "_")
        for p in candidate_paths
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
        threshold=0.95,
    )


import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction


class SpeechbrainEmbeddingFunction(EmbeddingFunction[list[bytes]]):
    """ChromaDB embedding function that accepts list of raw audio bytes and returns ECAPA-TDNN embeddings."""

    def __init__(self):
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(DATA_DIR / "pretrained_ecapa_tdnn"),
        )

    def __call__(self, input: list[bytes]) -> list[list[float]]:
        embeddings = []
        for audio_bytes in tqdm(input, desc="Extracting embeddings for Chroma"):
            emb = get_embedding(audio_bytes)  # Reuse existing robust get_embedding
            embeddings.append(emb.tolist())
        return embeddings


def demo_chroma_db() -> None:
    console.print(
        "[bold blue]Demo: ChromaDB persistent vector store for audio embeddings[/bold blue]"
    )

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
        base_bytes,  # 0
        exact_dup,  # 1
        base_bytes,  # 2
        noisy_bytes,  # 3
        diff_bytes,  # 4
    ]

    ids = [f"clip_{i:03d}" for i in range(len(audio_bytes_list))]
    metadatas = [
        {"label": "original_speech", "type": "speech"},
        {"label": "exact_duplicate", "type": "duplicate"},
        {"label": "original_speech_again", "type": "speech"},
        {"label": "noisy_version", "type": "near_duplicate"},
        {"label": "silence_different", "type": "different"},
    ]

    # Initialize Chroma client (persistent)
    persist_dir = DATA_DIR / "chroma_persistent"
    client = chromadb.PersistentClient(path=str(persist_dir))

    embedding_fn = SpeechbrainEmbeddingFunction()

    collection_name = "audio_embeddings_demo"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={
            "hnsw:space": "cosine"
        },  # Use cosine distance (critical for speaker embeddings)
    )

    console.print(
        f"[cyan]Adding {len(audio_bytes_list)} audio clips to collection '{collection_name}'...[/cyan]"
    )

    # Check for duplicate entries and only add new IDs
    existing_ids = []
    try:
        get_result = collection.get(ids=ids)
        existing_ids = get_result.get("ids", []) if get_result else []
    except Exception:
        # If collection.get fails (e.g., collection is empty), treat as no existing ids
        existing_ids = []
    new_ids = [id_ for id_ in ids if id_ not in existing_ids]

    if not new_ids:
        console.print(
            "[yellow]All clips already exist in collection – skipping add (duplicate prevention)[/yellow]"
        )
    else:
        new_audio_bytes = [audio_bytes_list[ids.index(id_)] for id_ in new_ids]
        new_metadatas = [metadatas[ids.index(id_)] for id_ in new_ids]

        collection.add(
            ids=new_ids,
            embeddings=embedding_fn(new_audio_bytes),
            metadatas=new_metadatas,
            documents=new_ids,
        )
        console.print(
            f"[green]Added {len(new_ids)} new clip(s) to persistent DB[/green]"
        )

    console.print(f"[green]Collection now has {collection.count()} documents[/green]")
    console.print(f"[green]Persistent database at:[/green] {persist_dir}")

    # Query with the noisy version
    query_bytes = noisy_bytes

    console.print(
        "[cyan]Querying collection with noisy clip (should match originals + exact dups highly)...[/cyan]"
    )
    results = collection.query(
        query_embeddings=embedding_fn([query_bytes]),
        n_results=5,
        include=["metadatas", "documents", "distances"],
    )

    # Print results table
    table = Table(title="ChromaDB Query Results (noisy clip query)")
    table.add_column("Rank", justify="right")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Type")
    table.add_column("Distance", justify="right")
    table.add_column("Cosine Similarity", justify="right")

    for rank, (id_, meta, dist) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0]), 1
    ):
        similarity = (
            1 - dist
        )  # Correct conversion for cosine distance (range 0–2 → similarity 1–-1)
        table.add_row(
            str(rank),
            id_,
            meta["label"],
            meta["type"],
            f"{dist:.4f}",
            f"{similarity:+.4f}",
        )

    console.print(table)

    # Highlight high-confidence matches
    high_match_ids = [
        id_
        for id_, dist in zip(results["ids"][0], results["distances"][0])
        if (1 - dist) > 0.95
    ]
    console.print(
        f"[bold green]High-confidence matches (>0.95 cosine):[/bold green] {high_match_ids}"
    )

    # Save demo info
    demo_results = {
        "demo_name": "chromadb_persistent_demo",
        "timestamp": datetime.now().isoformat(),
        "collection_name": collection_name,
        "persist_directory": str(persist_dir),
        "total_documents": collection.count(),
        "query_description": "noisy version of original speech clip",
        "top_matches": [
            {"id": id_, "label": meta["label"], "cosine_similarity": round(1 - dist, 6)}
            for id_, meta, dist in zip(
                results["ids"][0], results["metadatas"][0], results["distances"][0]
            )
        ],
    }

    demo_dir = RESULTS_DIR / "demo_chromadb"
    demo_dir.mkdir(parents=True, exist_ok=True)
    json_path = demo_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2)

    console.print(f"[green]ChromaDB demo results saved to:[/green] {json_path}")
    console.print(f"[green]Persistent database available at:[/green] {persist_dir}")
    console.print(
        "[yellow]Note: You can reload this DB later with PersistentClient(path=...) and get_collection('audio_embeddings_demo')[/yellow]"
    )


def demo_subsegment_search():
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    from jet.audio.utils import extract_audio_segment

    console.print(
        "[bold blue]Demo: Sub-segment search - query from 0.2-0.8 s inside segment_0002[/bold blue]"
    )

    candidate_paths = AUDIO_DIR
    query_path = Path(QUERY_PATH)

    if not query_path.exists():
        console.print(f"[red]Query source file not found:[/red] {query_path}")
        return

    query_waveform, sr = extract_audio_segment(str(query_path), start=0.2, end=0.8)

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

    console.print(
        f"[cyan]Query duration:[/cyan] {query_duration:.3f}s (0.2-0.8 s of segment_0002)"
    )
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
        label = str(cand_path.parent.name) + "_" + cand_path.name.replace(".", "_")

        # Handle short segments (pad if needed)
        if total_samples < window_samples:
            padding = window_samples - total_samples
            window_padded = torch.nn.functional.pad(
                waveform.unsqueeze(0), (0, padding)
            ).squeeze(0)
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
            if start_sample < total_samples and total_samples - start_sample >= int(
                sr * 0.3
            ):
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

        results.append(
            {
                "candidate": label,
                "duration_sec": round(duration_sec, 3),
                "best_window_start_sec": round(local_best_start, 3),
                "best_window_end_sec": round(local_best_start + window_sec, 3),
                "max_cosine_score": round(local_max_score, 4),
            }
        )

    if not results:
        console.print("[red]No candidates processed.[/red]")
        return

    results.sort(key=lambda x: x["max_cosine_score"], reverse=True)

    table = Table(
        title="Top matching sub-segments (query = 0.2-0.8 s inside segment_0002)"
    )
    table.add_column("Rank", justify="right")
    table.add_column("Candidate File")
    table.add_column("Duration (s)")
    table.add_column("Best Window (sec)")
    table.add_column("Cosine Score", justify="right")
    for rank, res in enumerate(results, 1):
        window_str = f"{res['best_window_start_sec']} - {res['best_window_end_sec']}"
        table.add_row(
            str(rank),
            res["candidate"],
            f"{res['duration_sec']:.2f}",
            window_str,
            f"{res['max_cosine_score']:.4f}",
        )
    console.print(table)

    # Direct comparison of query vs absolute best-matching sub-segment
    console.print(
        "\n[bold magenta]Direct comparison: Query (0.2-0.8 s) vs Best-matching sub-segment[/bold magenta]"
    )

    if best_matching_window_bytes is None:
        console.print("[red]No matching window found.[/red]")
    else:
        best_window_emb = get_embedding(best_matching_window_bytes)
        final_score = cosine_similarity(query_emb, best_window_emb)

        top_window_str = (
            f"{best_window_start:.3f} - {best_window_start + window_sec:.3f} s"
        )

        console.print("Query: 0.200 - 0.800 s from segment_0002")
        console.print(f"↔ Best sub-segment: {best_candidate_label} @ {top_window_str}")
        console.print(f"[bold green]Cosine similarity:[/bold green] {final_score:.6f}")

        if abs(final_score - 1.0) < 1e-4:
            console.print(
                "[bold green]✓ Perfect match - the window exactly corresponds to the query interval[/bold green]"
            )
        elif final_score > 0.85:
            console.print(
                "[bold cyan]✓ Very strong match (near-identical content)[/bold cyan]"
            )
        else:
            console.print(
                "[yellow]Moderate match - short duration limits embedding reliability[/yellow]"
            )

    # In demo_subsegment_search(), before save_demo_results()
    results_data = {
        "query_interval": "0.200 - 0.800 s from segment_0002",
        "top_matches": [
            {
                "rank": r,
                "candidate": res["candidate"],
                "duration_sec": res["duration_sec"],
                "best_window": f"{res['best_window_start_sec']} - {res['best_window_end_sec']}",
                "score": res["max_cosine_score"],
            }
            for r, res in enumerate(results, 1)
        ],
        "direct_comparison": {
            "query": "0.200 - 0.800 s",
            "best_subsegment": f"{best_candidate_label} @ {best_window_start:.3f} - {best_window_start + window_sec:.3f} s",
            "cosine_score": round(best_score, 6),
            "verdict": "Very strong match" if best_score > 0.85 else "Moderate",
        },
    }

    # Then update save_demo_results() call:
    save_demo_results(
        demo_name="subsegment_search_inner_partial_0.2-0.8s",
        sim_matrix=results_data,  # Or add as new key
        labels=[r["candidate"] for r in results],
        groups=[],
        threshold=None,
    )


def demo_subsegment_search_with_adaptive_window_growth():
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    from jet.audio.utils import extract_audio_segment

    console.print(
        "[bold blue]Demo: Sub-segment search - adaptive_window_growth[/bold blue]"
    )

    candidate_paths = AUDIO_DIR
    query_path = Path(QUERY_PATH)

    if not query_path.exists():
        console.print(f"[red]Query source file not found:[/red] {query_path}")
        return

    query_waveform, sr = extract_audio_segment(
        str(query_path),
        start=0.0,
        end=None,
    )

    query_audio = np.asarray(query_waveform, dtype=np.float32)

    if query_audio.size == 0:
        raise RuntimeError("Query audio is empty after extraction")

    if query_audio.ndim == 2:
        query_audio = query_audio.mean(axis=0)
    query_tensor = torch.from_numpy(query_audio).unsqueeze(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, query_tensor, sr)
        query_path = tmp.name

    query_bytes = Path(query_path).read_bytes()
    os.unlink(query_path)

    query_emb = get_embedding(query_bytes)
    query_duration = len(query_audio) / sr

    if query_duration < 0.25:
        raise RuntimeError(
            f"Query duration too short ({query_duration:.3f}s) for embedding search"
        )

    # Adaptive window: use query duration as window size
    window_sec: float = query_duration
    hop_sec: float = max(0.1, window_sec / 4)  # minimum 0.1 s hop for short queries
    window_samples = int(sr * window_sec)
    expand_step_sec = 0.1
    expand_step_samples = int(sr * expand_step_sec)

    if window_samples <= 0:
        raise RuntimeError("Computed window_samples is zero")

    grow_threshold = 0.92
    hop_samples = int(sr * hop_sec)

    console.print(
        f"[cyan]Query duration:[/cyan] {query_duration:.3f}s (0.2-0.8 s of segment_0002)"
    )
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
        label = str(cand_path.parent.name) + "_" + cand_path.name.replace(".", "_")

        # Handle short segments (pad if needed)
        if total_samples < window_samples:
            padding = window_samples - total_samples
            window_padded = torch.nn.functional.pad(
                waveform.unsqueeze(0), (0, padding)
            ).squeeze(0)
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
            if start_sample < total_samples and total_samples - start_sample >= int(
                sr * 0.3
            ):
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

        # --- Adaptive window growth around best local window ---
        best_start_sample = int(local_best_start * sr)
        best_end_sample = best_start_sample + window_samples

        grown_start = best_start_sample
        grown_end = best_end_sample
        grown_score = local_max_score

        while True:
            expanded = False

            # Try expanding left
            if grown_start - expand_step_samples >= 0:
                cand_start = grown_start - expand_step_samples
                cand_win = waveform[cand_start:grown_end]
                buf = io.BytesIO()
                torchaudio.save(buf, cand_win.unsqueeze(0), sr, format="wav")
                score = cosine_similarity(query_emb, get_embedding(buf.getvalue()))
                if score >= grow_threshold:
                    grown_start = cand_start
                    grown_score = score
                    expanded = True

            # Try expanding right
            if grown_end + expand_step_samples <= total_samples:
                cand_end = grown_end + expand_step_samples
                cand_win = waveform[grown_start:cand_end]
                buf = io.BytesIO()
                torchaudio.save(buf, cand_win.unsqueeze(0), sr, format="wav")
                score = cosine_similarity(query_emb, get_embedding(buf.getvalue()))
                if score >= grow_threshold:
                    grown_end = cand_end
                    grown_score = score
                    expanded = True

            if not expanded:
                break

        grown_start_sec = grown_start / sr
        grown_end_sec = grown_end / sr

        local_best_start = grown_start_sec
        local_best_window_bytes = None
        local_max_score = grown_score

        results.append(
            {
                "candidate": label,
                "duration_sec": round(duration_sec, 3),
                "best_window_start_sec": round(grown_start_sec, 3),
                "best_window_end_sec": round(grown_end_sec, 3),
                "max_cosine_score": round(local_max_score, 4),
            }
        )

    if not results:
        console.print("[red]No candidates processed.[/red]")
        return

    results.sort(key=lambda x: x["max_cosine_score"], reverse=True)

    table = Table(
        title="Top matching sub-segments (query = 0.2-0.8 s inside segment_0002)"
    )
    table.add_column("Rank", justify="right")
    table.add_column("Candidate File")
    table.add_column("Duration (s)")
    table.add_column("Best Window (sec)")
    table.add_column("Cosine Score", justify="right")
    for rank, res in enumerate(results, 1):
        window_str = f"{res['best_window_start_sec']} - {res['best_window_end_sec']}"
        table.add_row(
            str(rank),
            res["candidate"],
            f"{res['duration_sec']:.2f}",
            window_str,
            f"{res['max_cosine_score']:.4f}",
        )
    console.print(table)

    # Direct comparison of query vs absolute best-matching sub-segment
    console.print(
        "\n[bold magenta]Direct comparison: Query (0.2-0.8 s) vs Best-matching sub-segment[/bold magenta]"
    )

    if best_matching_window_bytes is None:
        console.print("[red]No matching window found.[/red]")
    else:
        best_window_emb = get_embedding(best_matching_window_bytes)
        final_score = cosine_similarity(query_emb, best_window_emb)

        top_window_str = (
            f"{best_window_start:.3f} - {best_window_start + window_sec:.3f} s"
        )

        console.print("Query: 0.200 - 0.800 s from segment_0002")
        console.print(f"↔ Best sub-segment: {best_candidate_label} @ {top_window_str}")
        console.print(f"[bold green]Cosine similarity:[/bold green] {final_score:.6f}")

        if abs(final_score - 1.0) < 1e-4:
            console.print(
                "[bold green]✓ Perfect match - the window exactly corresponds to the query interval[/bold green]"
            )
        elif final_score > 0.85:
            console.print(
                "[bold cyan]✓ Very strong match (near-identical content)[/bold cyan]"
            )
        else:
            console.print(
                "[yellow]Moderate match - short duration limits embedding reliability[/yellow]"
            )

    # In demo_subsegment_search(), before save_demo_results()
    results_data = {
        "query_interval": "0.200 - 0.800 s from segment_0002",
        "top_matches": [
            {
                "rank": r,
                "candidate": res["candidate"],
                "duration_sec": res["duration_sec"],
                "best_window": f"{res['best_window_start_sec']} - {res['best_window_end_sec']}",
                "score": res["max_cosine_score"],
            }
            for r, res in enumerate(results, 1)
        ],
        "direct_comparison": {
            "query": "0.200 - 0.800 s",
            "best_subsegment": f"{best_candidate_label} @ {best_window_start:.3f} - {best_window_start + window_sec:.3f} s",
            "cosine_score": round(best_score, 6),
            "verdict": "Very strong match" if best_score > 0.85 else "Moderate",
        },
    }

    # Then update save_demo_results() call:
    save_demo_results(
        demo_name="subsegment_search_inner_partial_0.2-0.8s",
        sim_matrix=results_data,  # Or add as new key
        labels=[r["candidate"] for r in results],
        groups=[],
        threshold=None,
    )


def demo_subsegment_search_with_dtw_refinement():
    """
    Demo:
    1) Coarse subsegment localization using ECAPA embeddings (sliding window)
    2) DTW refinement on log-mel features for precise alignment
    """
    import io
    from pathlib import Path

    import librosa
    import librosa.sequence
    import numpy as np
    from jet.audio.utils import extract_audio_segment

    console.print("[bold blue]Demo: Sub-segment search with DTW refinement[/bold blue]")

    candidate_paths = AUDIO_DIR
    query_path = Path(QUERY_PATH)

    if not query_path.exists():
        console.print(f"[red]Query file not found:[/red] {query_path}")
        return

    query_waveform, sr = extract_audio_segment(str(query_path))
    query_audio = np.asarray(query_waveform, dtype=np.float32)
    if query_audio.size == 0:
        raise RuntimeError("Query audio is empty after extraction")
    if query_audio.ndim == 2:
        query_audio = query_audio.mean(axis=0)

    query_tensor = torch.from_numpy(query_audio).unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, query_tensor, sr, format="wav")
    query_bytes = buf.getvalue()

    query_emb = get_embedding(query_bytes)
    query_duration = len(query_audio) / sr

    if query_duration < 0.25:
        raise RuntimeError(
            f"Query too short for DTW refinement ({query_duration:.3f}s)"
        )

    window_sec = query_duration
    hop_sec = max(0.1, window_sec / 4)
    window_samples = int(sr * window_sec)
    hop_samples = int(sr * hop_sec)

    coarse_results = []

    console.print(f"[cyan]Query duration:[/cyan] {query_duration:.3f}s")

    # -------------------------
    # Stage 1: Coarse search
    # -------------------------
    for cand_path in tqdm(candidate_paths, desc="Coarse embedding search"):
        waveform, file_sr = torchaudio.load(cand_path)
        if file_sr != sr:
            waveform = torchaudio.functional.resample(waveform, file_sr, sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.squeeze(0)
        total_samples = waveform.shape[0]

        best_score = -1.0
        best_start = 0

        for start in range(0, max(1, total_samples - window_samples), hop_samples):
            end = start + window_samples
            if end > total_samples:
                break

            win = waveform[start:end]
            buf = io.BytesIO()
            torchaudio.save(buf, win.unsqueeze(0), sr, format="wav")
            score = cosine_similarity(query_emb, get_embedding(buf.getvalue()))

            if score > best_score:
                best_score = score
                best_start = start

        coarse_results.append(
            {"path": cand_path, "start_sample": best_start, "score": best_score}
        )

    coarse_results.sort(key=lambda x: x["score"], reverse=True)
    top = coarse_results[0]

    console.print(
        f"[green]Top coarse match:[/green] "
        f"{Path(top['path']).name} @ {top['start_sample'] / sr:.3f}s "
        f"(cosine={top['score']:.4f})"
    )

    # -----------------------------------
    # Stage 2: DTW refinement
    # -----------------------------------

    refine_margin_sec = 0.5
    refine_start = max(0, top["start_sample"] - int(refine_margin_sec * sr))

    # Load candidate for DTW
    cand_waveform, _ = torchaudio.load(top["path"])
    if cand_waveform.shape[0] > 1:
        cand_waveform = cand_waveform.mean(0, keepdim=True)
    cand_waveform = cand_waveform.squeeze(0).numpy()
    total_samples = cand_waveform.shape[0]
    refine_end = min(
        refine_start + int((query_duration + 2 * refine_margin_sec) * sr), total_samples
    )
    cand_region = cand_waveform[refine_start:refine_end]

    # Log-mel features
    query_mel = librosa.feature.melspectrogram(
        y=query_audio, sr=sr, n_mels=40, hop_length=160, n_fft=400
    )
    cand_mel = librosa.feature.melspectrogram(
        y=cand_region, sr=sr, n_mels=40, hop_length=160, n_fft=400
    )

    query_mel_db = librosa.power_to_db(query_mel, ref=np.max)
    cand_mel_db = librosa.power_to_db(cand_mel, ref=np.max)

    if not np.isfinite(query_mel_db).all() or not np.isfinite(cand_mel_db).all():
        raise RuntimeError("NaNs detected in mel-spectrograms; aborting DTW")

    D, wp = librosa.sequence.dtw(X=query_mel_db, Y=cand_mel_db, metric="cosine")

    path_len = len(wp)
    dtw_cost = D[-1, -1] / max(1, path_len)

    # Frame → time
    hop_length = 160
    cand_frames = [p[1] for p in wp]
    cand_start_frame = min(cand_frames)
    cand_end_frame = max(cand_frames)

    refined_start_sec = (refine_start + cand_start_frame * hop_length) / sr
    refined_end_sec = (refine_start + cand_end_frame * hop_length) / sr

    console.print("\n[bold magenta]DTW refinement result[/bold magenta]")
    console.print(f"Candidate file: {Path(top['path']).name}")
    console.print(f"Refined match: {refined_start_sec:.3f} – {refined_end_sec:.3f} s")
    console.print(f"DTW normalized cost: {dtw_cost:.6f}")

    verdict = (
        "Excellent alignment"
        if dtw_cost < 0.15
        else "Good alignment"
        if dtw_cost < 0.30
        else "Weak / partial alignment"
    )

    console.print(f"[bold green]Verdict:[/bold green] {verdict}")


if __name__ == "__main__":
    demo_exact_duplicates()
    console.print("\n" + "=" * 80 + "\n")
    demo_random_short_clips()
    console.print("\n" + "=" * 80 + "\n")
    demo_local_files()
    console.print("\n" + "=" * 80 + "\n")
    demo_subsegment_search()
    console.print("\n" + "=" * 80 + "\n")
    demo_subsegment_search_with_adaptive_window_growth()
    console.print("\n" + "=" * 80 + "\n")
    demo_subsegment_search_with_dtw_refinement()
    console.print("\n" + "=" * 80 + "\n")
    demo_chroma_db()
