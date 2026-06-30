import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from jet.audio.audio_waveform.vad.vad_logging import linkify
from jet.audio.helpers.config import FRAME_SHIFT_MS, SAMPLE_RATE
from jet.audio.speech.vad_extractors import (
    extract_trough_to_trough,
)
from jet.audio.utils.loader import load_audio
from rich.console import Console

console = Console()
DEFAULT_AUDIO = "/Users/jethroestrada/.cache/files/audio/recording_3_speakers.wav"
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
parser = argparse.ArgumentParser(
    description="Analyze VAD speech/voice probabilities and find peaks/troughs"
)
parser.add_argument(
    "audio_path",
    nargs="?",
    default=DEFAULT_AUDIO,
    help=(
        "Path to either:\n"
        "- JSON file with speech probabilities\n"
        "- Audio file (wav/mp3/flac/etc.) to run VAD on\n"
        "If not provided, uses a sample sequence."
    ),
)
parser.add_argument(
    "--output-dir",
    "-o",
    default=str(OUTPUT_DIR),
    help="Output directory for generated files (default: ./generated/<script name>)",
)
parser.add_argument(
    "--min-duration",
    "-d",
    type=float,
    default=0.25,
    help="Minimum segment duration in seconds (default: 0.25)",
)
args = parser.parse_args()
frame_offset = 0
smoothing_window = 0
min_duration_s = args.min_duration
output_dir = Path(args.output_dir)
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)
console.print(f"[cyan]min_duration_s={min_duration_s:.3f}s[/cyan]")
audio_np, _ = load_audio(args.audio_path, sr=SAMPLE_RATE, mono=True)
segments_with_audio, probs = extract_trough_to_trough(
    probs_or_audio=audio_np,
    frame_shift_ms=FRAME_SHIFT_MS,
    sample_rate=SAMPLE_RATE,
    with_audio=True,
    with_scores=True,
    min_duration_s=min_duration_s,
)
segs_out_dir = output_dir / "trough_to_trough"
segs_out_dir.mkdir(parents=True, exist_ok=True)
all_segments_meta = []
# ─── Get probs array for plotting ───
probs_array = np.array(probs, dtype=float) if probs else np.array([])
n_frames = len(probs_array)
for idx, (seg_meta, audio_slice) in enumerate(segments_with_audio):
    seg_dir = segs_out_dir / f"segment_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(seg_meta, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(seg_meta)
    if len(audio_slice) > 0:
        sf.write(str(seg_dir / "sound.wav"), audio_slice, SAMPLE_RATE)
    if seg_meta.get("segment_probs"):
        with open(seg_dir / "probs.json", "w", encoding="utf-8") as fh:
            json.dump(seg_meta["segment_probs"], fh, ensure_ascii=False, indent=2)
        if seg_meta.get("prob_stats"):
            probs_info = {
                "frame_shift_sec": FRAME_SHIFT_MS / 1000.0,
                "frame_start": seg_meta["start_frame"],
                "frame_end": seg_meta["end_frame"],
                "stats": seg_meta["prob_stats"],
            }
            with open(seg_dir / "probs_info.json", "w", encoding="utf-8") as fh:
                json.dump(probs_info, fh, ensure_ascii=False, indent=2)

    # ─── END UPDATED ───
    # ─── Generate plot.png for this segment ───
    if n_frames > 0:
        f_start = max(0, seg_meta["start_frame"])
        f_end = min(n_frames, seg_meta["end_frame"] + 1)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)
        ax.axvspan(f_start, f_end, alpha=0.12, color="purple", label="trough span")
        is_first = idx == 0
        ax.axvline(
            x=seg_meta["start_frame"],
            color="gray" if is_first else "red",
            linestyle="--",
            linewidth=1.5,
            label=f"{'origin' if is_first else 'start trough'} (frame {seg_meta['start_frame']})",
        )
        if not is_first and seg_meta["start_frame"] < n_frames:
            ax.plot(
                seg_meta["start_frame"],
                probs_array[seg_meta["start_frame"]],
                "ro",
                markersize=9,
            )
        is_last = idx == len(segments_with_audio) - 1
        ax.axvline(
            x=seg_meta["end_frame"],
            color="gray" if is_last else "red",
            linestyle="--",
            linewidth=1.5,
            label=f"{'end of audio' if is_last else 'end trough'} (frame {seg_meta['end_frame']})",
        )
        if not is_last and seg_meta["end_frame"] < n_frames:
            ax.plot(
                seg_meta["end_frame"],
                probs_array[seg_meta["end_frame"]],
                "ro",
                markersize=9,
            )
        # Overlay speech segments as green spans if available
        if seg_meta.get("segments"):
            for speech_seg in seg_meta["segments"]:
                speech_start_s = speech_seg.get("start", 0)
                speech_end_s = speech_seg.get("end", 0)
                speech_start_frame = int(speech_start_s / (FRAME_SHIFT_MS / 1000.0))
                speech_end_frame = int(speech_end_s / (FRAME_SHIFT_MS / 1000.0))
                ax.axvspan(
                    speech_start_frame,
                    speech_end_frame,
                    alpha=0.15,
                    color="green",
                    label="speech" if speech_seg == seg_meta["segments"][0] else None,
                )
        ax.set_title(
            f"trough_to_trough · segment {idx:03d}  "
            f"[{seg_meta['start_s']:.3f} s – {seg_meta['end_s']:.3f} s]",
            fontsize=12,
        )
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="upper right")
        plt.tight_layout()
        plot_path = seg_dir / "plot.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()
        console.print(
            f"  [dim]Segment {idx:03d}:[/dim] "
            f"duration: {seg_meta['duration_s']:.2f}s | "
            f"frames {f_start}-{f_end} | {len(zoomed)} count"
        )

summary_path = output_dir / "trough_to_trough.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)
console.print(f"[bold green]Segments saved under:[/bold green] {linkify(segs_out_dir)}")
console.print(
    f"[bold green]Trough to trough summary saved to:[/bold green] {linkify(summary_path)}"
)
