# diarize_ultra_insights_fixed_and_final.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import numpy as np
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pyannote.core import SlidingWindowFeature, Segment
import jet


pipeline: Pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
)

def diarize_with_ultra_insights(audio_path: str | Path) -> DiarizeOutput:

    pipeline.instantiate({
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {"threshold": 0.60, "min_cluster_size": 5},
    })

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline.to(device)

    def save_artifact(step_name: str, artifact: Any, **kwargs):
        setattr(pipeline, f"_cached_{step_name}", artifact)

    file = {"uri": Path(audio_path).stem, "audio": str(audio_path)}
    output: DiarizeOutput = pipeline.apply(file, hook=save_artifact)

    return output


def print_ultra_insights(output: DiarizeOutput, audio_path: Path):
    rprint(Panel(f"[bold magenta]ULTRA INSIGHTS • {audio_path.name}[/]", expand=False))

    # 1. Final diarization with proper per-turn confidence
    rprint("\n[bold green]Final Diarization (with per-turn confidence)[/]")
    table = Table("Speaker", "Start", "End", "Duration", "Confidence")
    
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        # CORRECT WAY: intersect turn with speaker's total support
        speaker_support = output.speaker_diarization.label_support(speaker)  # → Timeline
        turn_segment = Segment(turn.start, turn.end)
        overlap_duration = speaker_support.crop(turn_segment).duration()

        # Confidence = how much of this turn is actually supported by the speaker
        confidence = overlap_duration / turn.duration if turn.duration > 0 else 0.0

        table.add_row(
            speaker,
            f"{turn.start:.3f}s",
            f"{turn.end:.3f}s",
            f"{turn.duration:.3f}s",
            f"{confidence:.3f}",
        )
    rprint(table)

    # 2. Speaker similarity matrix
    if output.speaker_embeddings is not None and len(output.speaker_embeddings) >= 2:
        from torch.nn.functional import cosine_similarity
        emb = torch.from_numpy(output.speaker_embeddings)
        sim = cosine_similarity(emb[:, None], emb[None, :], dim=-1).numpy()

        rprint("\n[bold yellow]Speaker Similarity (Cosine)[/]")
        sim_table = Table("", *[f"[bold]{l}[/]" for l in output.speaker_diarization.labels()])
        for i, row in enumerate(sim):
            sim_table.add_row(output.speaker_diarization.labels()[i], *[f"{v:.3f}" for v in row])
        rprint(sim_table)

    # 3. VAD stats
    seg: SlidingWindowFeature | None = getattr(pipeline, "_cached_segmentation", None)
    if seg is not None:
        speech_frames = (seg.data > 0.5).sum()
        total_frames = seg.data.shape[0] * seg.data.shape[1]  # chunks × frames
        rprint("\n[bold]Voice Activity Detection[/]")
        rprint(f"   • Speech frames     : {speech_frames:,} / {total_frames:,}")
        rprint(f"   • Speech ratio      : {speech_frames / total_frames:.1%}")
        rprint(f"   • Avg speech prob   : {seg.data.mean():.3f}")

    # 4. Instantaneous speaker count
    count: SlidingWindowFeature | None = getattr(pipeline, "_cached_speaker_counting", None)
    if count is not None:
        counts = np.bincount(count.data.flatten().astype(int))
        rprint("\n[bold]Instantaneous Active Speakers (per frame)[/]")
        count_table = Table("Speakers", "Frames", "%")
        for n, c in enumerate(counts):
            if c > 0:
                count_table.add_row(str(n), f"{c:,}", f"{c/len(count):.1%}")
        rprint(count_table)

    # 5. Clustering diagnostics
    hard_clusters = getattr(pipeline, "_cached_hard_clusters", None)
    if hard_clusters is not None:
        valid = hard_clusters[hard_clusters >= 0]
        cluster_sizes = np.bincount(valid) if len(valid) > 0 else np.array([])
        rprint("\n[bold cyan]Clustering Diagnostics[/]")
        rprint(f"   • Detected speakers     : {len(cluster_sizes)}")
        rprint(f"   • Cluster sizes (frames): {cluster_sizes.tolist()}")
        rprint(f"   • Threshold used        : {getattr(pipeline.clustering, 'threshold', 'N/A')}")

    # 6. Top-3 raw scores
    if seg is not None:
        rprint("\n[bold]Top-3 Raw Segmentation Scores per Speaker[/]")
        top_table = Table("Rank", "Time (s)", "Score", "Speaker")
        topk_indices = np.argpartition(-seg.data.reshape(-1, seg.data.shape[-1]), 3, axis=0)[:3]

        labels = list(output.speaker_diarization.labels())
        for spk_idx, spk in enumerate(labels):
            for rank, frame_idx in enumerate(topk_indices[:, spk_idx], 1):
                t = seg.sliding_window[frame_idx].middle
                score = seg.data.reshape(-1, seg.data.shape[-1])[frame_idx, spk_idx]
                top_table.add_row(str(rank), f"{t:.2f}", f"{score:.3f}", spk)
        rprint(top_table)


# ——— RUN ———
if __name__ == "__main__":
    audio_file = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251212_041845.wav")

    result = diarize_with_ultra_insights(audio_file)
    print_ultra_insights(result, audio_file)