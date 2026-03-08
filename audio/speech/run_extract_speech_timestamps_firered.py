# run_extract_speech_timestamps_speechbrain.py
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from jet.audio.speech.firered.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.utils import display_segments
from jet.file.utils import save_file
from jet.utils.text import format_sub_dir
from rich.console import Console
from silero_vad.utils_vad import read_audio

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def compute_speech_stats(
    segments: list[dict[str, Any]],
    include_non_speech: bool = False,
) -> dict[str, Any]:
    """Compute descriptive statistics from VAD segments.

    Focuses on speech segments; optionally reports non-speech too.
    Handles empty lists gracefully.
    """
    speech_segs = [s for s in segments if s.get("type") == "speech"]
    non_speech_segs = [s for s in segments if s.get("type") == "non-speech"]

    if not segments:
        return {
            "total_duration_sec": 0.0,
            "speech_duration_sec": 0.0,
            "speech_ratio": 0.0,
            "num_speech_segments": 0,
            "num_non_speech_segments": 0,
            "avg_speech_duration_sec": 0.0,
            "median_speech_duration_sec": 0.0,
            "min_speech_duration_sec": 0.0,
            "max_speech_duration_sec": 0.0,
            "avg_segment_prob": 0.0,
            "median_segment_prob": 0.0,
            "num_short_speech_fragments": 0,
        }

    total_duration_sec = max((s["end"] for s in segments), default=0.0)
    speech_durations = np.array([s["duration"] for s in speech_segs])
    speech_probs = (
        np.array([s["prob"] for s in speech_segs]) if speech_segs else np.array([])
    )

    speech_duration_sec = speech_durations.sum()

    stats = {
        "total_duration_sec": round(float(total_duration_sec), 3),
        "speech_duration_sec": round(float(speech_duration_sec), 3),
        "speech_ratio": (
            round(float(speech_duration_sec / total_duration_sec), 4)
            if total_duration_sec > 1e-6
            else 0.0
        ),
        "num_speech_segments": len(speech_segs),
        "num_non_speech_segments": len(non_speech_segs) if include_non_speech else 0,
    }

    if len(speech_durations) > 0:
        stats.update(
            {
                "avg_speech_duration_sec": round(float(speech_durations.mean()), 3),
                "median_speech_duration_sec": round(
                    float(np.median(speech_durations)), 3
                ),
                "min_speech_duration_sec": round(float(speech_durations.min()), 3),
                "max_speech_duration_sec": round(float(speech_durations.max()), 3),
                "num_short_speech_fragments": int((speech_durations < 0.5).sum()),
            }
        )
    if len(speech_probs) > 0:
        stats.update(
            {
                "avg_segment_prob": round(float(speech_probs.mean()), 4),
                "median_segment_prob": round(float(np.median(speech_probs)), 4),
            }
        )

    return stats


def save_segment_prob_plot(
    segment_probs: list[float],
    seg_start_sec: float,
    duration_sec: float,
    threshold: float,
    neg_threshold: float,
    save_path: Path,
    dpi: int = 120,
) -> None:
    """Save a simple line plot of speech probabilities for one segment."""
    if not segment_probs:
        return

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=dpi)
    times = [seg_start_sec + i * 0.01 for i in range(len(segment_probs))]  # ≈10 ms hop

    ax.plot(times, segment_probs, color="#1f77b4", lw=1.4, label="p(speech)")
    ax.axhline(
        threshold, color="#ff7f0e", ls="--", lw=1.1, label=f"thresh = {threshold}"
    )
    ax.axhline(
        neg_threshold,
        color="#d62728",
        ls=":",
        lw=1.1,
        label=f"neg_thresh = {neg_threshold}",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speech Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Speech Probability – {save_path.parent.name}")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def create_sub_dir(file: str):
    pass


def main(
    audio_file: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.5,  # activation_th
    neg_threshold: float = 0.25,  # deactivation_th
    min_silence_duration_sec: float = 0.250,
    min_speech_duration_sec: float = 0.250,
    max_speech_duration_sec: float = float("inf"),
    normalize_loudness: bool = False,
    include_non_speech: bool = False,
    apply_energy_VAD: bool = False,
):
    audio_file = str(audio_file)
    output_dir = Path(output_dir)
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments, all_speech_probs = extract_speech_timestamps(
        audio_file,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        return_seconds=True,
        # time_resolution=3,
        with_scores=True,
        # neg_threshold=neg_threshold,
        # normalize_loudness=normalize_loudness,
        include_non_speech=include_non_speech,  # already passed but ensure used
        # apply_energy_VAD=apply_energy_VAD,
    )
    waveform = read_audio(audio_file, sampling_rate=16000).unsqueeze(0)
    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments or [])}\n")
    if not segments:
        console.print(
            "[bold yellow]No speech segments detected – skipping save.[/bold yellow]"
        )
        return

    per_file_stats = compute_speech_stats(
        segments,
        include_non_speech=include_non_speech,
    )

    display_segments(segments)

    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        # seg_type = seg["type"]
        # if seg_type == "speech":
        #     type_color = "bold green"
        # else:
        #     type_color = "bold red"
        # console.print(
        #     f"[yellow][[/yellow] [bold white]{seg['start']:.2f}s[/bold white] - [bold white]{seg['end']:.2f}s[/bold white] [yellow]][/yellow] "
        #     f"duration=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
        #     f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
        #     f"type=[{type_color}]{seg_type}[/{type_color}]"
        # )
        folder_name = (
            f"segment_{seg['num']:03d}"  # use segment number for output subdir
        )
        seg_dir = segments_dir / folder_name
        seg_dir.mkdir(parents=True, exist_ok=True)
        save_file(seg, seg_dir / "segment.json", verbose=False)
        start_sample = int(seg["start"] * 16000)
        end_sample = int(seg["end"] * 16000)
        segment_audio = waveform[:, start_sample:end_sample]

        # ─── Save per-segment speech probability plot ───────────────────────
        plot_path = seg_dir / "speech_prob_curve.png"
        save_segment_prob_plot(
            seg["segment_probs"],
            seg["start"] if isinstance(seg["start"], float) else seg["start"] / 16000,
            seg["duration"],
            threshold,
            neg_threshold,
            plot_path,
        )

        torchaudio.save(
            str(seg_dir / "sound.wav"),
            segment_audio,
            sample_rate=16000,
            encoding="PCM_S",
            bits_per_sample=16,
        )
    save_file(segments, output_dir / "speech_timestamps.json")

    # Compute and save segment gaps

    if segments:
        gaps: list[dict[str, Any]] = []

        # Inter-segment silences: between consecutive speech segments
        for i in range(1, len(segments)):
            prev_end = segments[i - 1]["end"]
            curr_start = segments[i]["start"]
            gap_duration_val = curr_start - prev_end
            if gap_duration_val > 0:
                gaps.append(
                    {
                        "gap_idx": i,
                        "seg_idx_range": [i - 1, i],
                        "description": f"silence between speech segment {i - 1} and {i}",
                        "start": round(prev_end, 3),
                        "end": round(curr_start, 3),
                        "gap_duration": round(gap_duration_val, 3),
                    }
                )

        save_file(gaps, output_dir / "segment_gaps.json")
        save_file(all_speech_probs, output_dir / "all_speech_probs.json")


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    audio_paths = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_per_speech/last_5_mins.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav",
    ]

    include_non_speech = True
    normalize_loudness = False
    apply_energy_VAD = True

    threshold = 0.5
    neg_threshold = 0.25
    min_silence_duration_sec = 0.4
    min_speech_duration_sec = 0.5
    max_speech_duration_sec = 6.0

    summary: dict[str, Any] = {
        "total_files_processed": len(audio_paths),
        "files_with_speech": 0,
        "total_segments": 0,
        "per_file": defaultdict(dict),
    }
    for audio_path in audio_paths:
        sub_output_dir = OUTPUT_DIR / format_sub_dir(Path(audio_path).stem)
        Path(sub_output_dir).mkdir(parents=True, exist_ok=True)
        # --- run with per-file stats ---
        main(
            audio_path,
            sub_output_dir,
            normalize_loudness=normalize_loudness,
            include_non_speech=include_non_speech,
            threshold=threshold,
            neg_threshold=neg_threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            apply_energy_VAD=apply_energy_VAD,
        )
        per_file_stats = {}
        if (sub_output_dir / "speech_timestamps.json").exists():
            with open(sub_output_dir / "speech_timestamps.json") as f:
                segs = json.load(f)
            per_file_stats = compute_speech_stats(
                segs, include_non_speech=include_non_speech
            )
            # Rich stats replace simple count
            summary["files_with_speech"] += (
                1 if per_file_stats["num_speech_segments"] > 0 else 0
            )
            summary["total_segments"] += (
                per_file_stats["num_speech_segments"]
                + per_file_stats["num_non_speech_segments"]
            )
            summary["per_file"][str(audio_path)] = per_file_stats

        save_file(summary, Path(sub_output_dir) / "summary.json")
        console.print(
            f"\n[bold green]Global summary saved to:[/bold green] {Path(sub_output_dir) / 'summary.json'}"
        )
