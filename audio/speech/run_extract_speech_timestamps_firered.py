# run_extract_speech_timestamps_firered.py
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
from scipy.io import wavfile

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
    threshold: float = 0.5,
    neg_threshold: float = 0.25,
    min_silence_duration_sec: float = 0.250,
    min_speech_duration_sec: float = 0.250,
    max_speech_duration_sec: float = float("inf"),
    normalize_loudness: bool = False,
    include_non_speech: bool = False,
    apply_energy_VAD: bool = False,
    # New filters
    min_duration_sec: float = 0.0,  # -md / --min-duration
    min_prob: float = 0.0,  # -mp / --min-probs
):
    audio_file = str(audio_file)
    output_dir = Path(output_dir)
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")

    # Load ORIGINAL audio (native sample rate + channels preserved)
    orig_waveform, orig_sr = torchaudio.load(str(audio_file))
    console.print(
        f"[dim]Original audio: {orig_sr} Hz, {orig_waveform.shape[0]}ch[/dim]"
    )

    segments, all_speech_probs = extract_speech_timestamps(
        audio_file,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        return_seconds=True,
        with_scores=True,
        include_non_speech=include_non_speech,
    )

    if not segments:
        console.print(
            "[bold yellow]No speech segments detected – skipping save.[/bold yellow]"
        )
        return

    # === NEW: Apply min_duration and min_prob filtering ===
    original_count = len(segments)

    filtered_segments = []
    for seg in segments:
        duration_ok = seg["duration"] >= min_duration_sec
        prob_ok = seg["prob"] >= min_prob

        if duration_ok and prob_ok:
            filtered_segments.append(seg)
        else:
            console.print(
                f"[dim]Filtered out segment {seg['num']}: "
                f"dur={seg['duration']:.3f}s (min={min_duration_sec}), "
                f"prob={seg['prob']:.4f} (min={min_prob})[/dim]"
            )

    segments = filtered_segments

    # Renumber segments after filtering
    for i, seg in enumerate(segments, start=1):
        seg["num"] = i

    console.print(
        f"[bold green]Segments after filtering:[/bold green] {len(segments)} "
        f"(filtered {original_count - len(segments)})"
    )

    if not segments:
        console.print(
            "[bold yellow]No segments remained after filtering – skipping save.[/bold yellow]"
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
        folder_name = f"segment_{seg['num']:03d}"
        seg_dir = segments_dir / folder_name
        seg_dir.mkdir(parents=True, exist_ok=True)

        save_file(seg, seg_dir / "segment.json", verbose=False)

        # Slice from ORIGINAL audio using time-based timestamps
        start_sample = int(round(float(seg["start"]) * orig_sr))
        end_sample = int(round(float(seg["end"]) * orig_sr))
        end_sample = min(end_sample, orig_waveform.shape[1])

        segment_audio = orig_waveform[:, start_sample:end_sample]

        plot_path = seg_dir / "speech_prob_curve.png"
        save_segment_prob_plot(
            seg["segment_probs"],
            seg["start"] if isinstance(seg["start"], float) else seg["start"] / orig_sr,
            seg["duration"],
            threshold,
            neg_threshold,
            plot_path,
        )

        # Save WAV (int16)
        if segment_audio.shape[0] == 1:
            data = segment_audio.squeeze(0).numpy()
        else:
            data = segment_audio.numpy().T

        data_int16 = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(str(seg_dir / "sound.wav"), orig_sr, data_int16)

    save_file(segments, output_dir / "speech_timestamps.json")

    # Compute and save segment gaps (only between remaining segments)
    if len(segments) > 1:
        gaps: list[dict[str, Any]] = []
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
    import argparse
    import json
    import shutil
    from collections import defaultdict
    from pathlib import Path

    DEFAULT_AUDIO_PATH = (
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/"
        "audio/generated/run_record_mic/recording_3_speakers.wav"
    )

    parser = argparse.ArgumentParser(
        description="Process audio file for speech segmentation and VAD."
    )

    parser.add_argument(
        "audio_path",
        nargs="?",
        type=Path,
        default=Path(DEFAULT_AUDIO_PATH),
        help="Path to the audio file to process",
    )

    # Optional arguments with short flags
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Base output directory"
    )
    parser.add_argument(
        "-n",
        "--include-non-speech",
        action="store_true",
        default=True,
        help="Include non-speech segments (default: True)",
    )
    parser.add_argument(
        "-N",
        "--no-include-non-speech",
        dest="include_non_speech",
        action="store_false",
        help="Disable non-speech segments",
    )
    parser.add_argument(
        "-l",
        "--normalize-loudness",
        action="store_true",
        default=False,
        help="Normalize loudness (default: False)",
    )
    parser.add_argument(
        "-e",
        "--apply-energy-vad",
        action="store_true",
        default=True,
        help="Apply energy-based VAD (default: True)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Speech threshold (default: 0.5)",
    )
    parser.add_argument(
        "-T",
        "--neg-threshold",
        type=float,
        default=0.25,
        help="Non-speech threshold (default: 0.25)",
    )
    parser.add_argument(
        "-s",
        "--min-silence",
        type=float,
        default=0.2,
        help="Minimum silence duration in seconds (default: 0.2)",
    )
    parser.add_argument(
        "-m",
        "--min-speech",
        type=float,
        default=0.3,
        help="Minimum speech duration in seconds (default: 0.3)",
    )
    parser.add_argument(
        "-M",
        "--max-speech",
        type=float,
        default=6.0,
        help="Maximum speech duration in seconds (default: 6.0)",
    )

    # New filtering arguments
    parser.add_argument(
        "-md",
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Minimum segment duration in seconds to keep (default: 0.0 = no filter)",
    )
    parser.add_argument(
        "-mp",
        "--min-prob",
        type=float,
        default=0.0,
        help="Minimum average probability / score to keep segment (default: 0.0 = no filter)",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    else:
        OUTPUT_DIR = args.output_dir

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = args.audio_path
    sub_output_dir = OUTPUT_DIR / format_sub_dir(audio_path.stem)
    sub_output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "total_files_processed": 1,
        "files_with_speech": 0,
        "total_segments": 0,
        "per_file": defaultdict(dict),
    }

    # Run main with new filters
    main(
        str(audio_path),
        sub_output_dir,
        normalize_loudness=args.normalize_loudness,
        include_non_speech=args.include_non_speech,
        threshold=args.threshold,
        neg_threshold=args.neg_threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        apply_energy_VAD=args.apply_energy_vad,
        min_duration_sec=args.min_duration_sec,
        min_prob=args.min_prob,
    )

    # Stats saving (unchanged)
    per_file_stats = {}
    speech_json = sub_output_dir / "speech_timestamps.json"
    if speech_json.exists():
        with open(speech_json) as f:
            segs = json.load(f)

        per_file_stats = compute_speech_stats(
            segs, include_non_speech=args.include_non_speech
        )

        summary["files_with_speech"] = (
            1 if per_file_stats.get("num_speech_segments", 0) > 0 else 0
        )
        summary["total_segments"] = per_file_stats.get(
            "num_speech_segments", 0
        ) + per_file_stats.get("num_non_speech_segments", 0)
        summary["per_file"][str(audio_path)] = per_file_stats

    save_file(summary, sub_output_dir / "summary.json")

    console.print(
        f"\n[bold green]Processing complete! Summary saved to:[/bold green] {sub_output_dir / 'summary.json'}"
    )
