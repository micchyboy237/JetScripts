import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from jet.audio.audio_waveform.vad.vad_logging import linkify
from jet.audio.helpers.config import FRAME_SHIFT_MS, SAMPLE_RATE
from jet.audio.normalization.dtype_conversion import convert_audio_dtype
from jet.audio.speech.vad_extractors import extract_peaks, load_probs
from jet.audio.utils.info import display_audio_info
from rich.console import Console

console = Console()
DEFAULT_AUDIO = "/Users/jethroestrada/.cache/files/audio/recording_3_speakers.wav"
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
parser = argparse.ArgumentParser(
    description="Extract VAD probability peaks (local maxima) from audio"
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
    "--height",
    type=float,
    default=None,
    help="Minimum probability for a peak (e.g. 0.6). If not set, auto-computed.",
)
parser.add_argument(
    "--distance",
    type=int,
    default=None,
    help="Minimum frames between peaks (e.g. 5-20).",
)
parser.add_argument(
    "--prominence",
    type=float,
    default=None,
    help="Required prominence of peaks (e.g. 0.1-0.3).",
)
parser.add_argument(
    "--width",
    type=int,
    default=None,
    help="Minimum width of peaks in frames.",
)
parser.add_argument(
    "--context-window",
    "-c",
    type=float,
    default=0.1,
    help="Context window in seconds around each peak for audio extraction (default: 0.1)",
)
parser.add_argument(
    "--quantize",
    "-q",
    action="store_true",
    help="Quantize audio to int16 before processing (default: False)",
)
args = parser.parse_args()
context_window_s = args.context_window
output_dir = Path(args.output_dir)
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)
console.print(
    f"[cyan]height={args.height}, distance={args.distance}, prominence={args.prominence}, width={args.width}[/cyan]"
)
console.print(f"[cyan]context_window_s={context_window_s:.3f}s[/cyan]")
probs, audio_np = load_probs(args.audio_path)
if args.quantize:
    audio_np = convert_audio_dtype(audio_np, "int16")
display_audio_info(audio_np)
peaks = extract_peaks(
    probs_or_audio=probs,
    frame_shift_ms=FRAME_SHIFT_MS,
    height=args.height,
    distance=args.distance,
    prominence=args.prominence,
    width=args.width,
)
# Save peaks as individual segment directories with audio, probs, and plots
segs_out_dir = output_dir / "peaks"
segs_out_dir.mkdir(parents=True, exist_ok=True)
all_segments_meta = []
probs_array = np.array(probs, dtype=float) if probs else np.array([])
n_frames = len(probs_array)
total_audio_samples = len(audio_np) if audio_np is not None else 0
context_frames = int(context_window_s / (FRAME_SHIFT_MS / 1000.0))

for idx, peak in enumerate(peaks):
    seg_dir = segs_out_dir / f"peak_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Save peak metadata
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(peak, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(peak)

    # Extract audio context around the peak
    if audio_np is not None and total_audio_samples > 0:
        start_s = max(0, peak["start_s"] - context_window_s)
        end_s = min(total_audio_samples / SAMPLE_RATE, peak["end_s"] + context_window_s)
        start_sample = int(start_s * SAMPLE_RATE)
        end_sample = int(end_s * SAMPLE_RATE)
        audio_slice = audio_np[start_sample:end_sample]
        if len(audio_slice) > 0:
            sf.write(str(seg_dir / "sound.wav"), audio_slice, SAMPLE_RATE)
            console.print(
                f"  [dim]Peak {idx:03d}:[/dim] audio context [{start_s:.3f}s - {end_s:.3f}s] "
                f"({len(audio_slice)} samples, {len(audio_slice) / SAMPLE_RATE:.3f}s)"
            )

    # Save peak probability details
    if "details" in peak:
        probs_info = {
            "frame_shift_sec": FRAME_SHIFT_MS / 1000.0,
            "peak_frame": peak["frame_start"],
            "peak_probability": peak["details"].get("peak_probability"),
            "prominence": peak["details"].get("prominence"),
            "width": peak["details"].get("width"),
            "left_base": peak["details"].get("left_base"),
            "right_base": peak["details"].get("right_base"),
        }
        with open(seg_dir / "probs_info.json", "w", encoding="utf-8") as fh:
            json.dump(probs_info, fh, ensure_ascii=False, indent=2)

    # Generate probability plot centered on this peak with context
    if n_frames > 0:
        peak_frame = peak["frame_start"]
        f_start = max(0, peak_frame - context_frames * 2)
        f_end = min(n_frames, peak_frame + context_frames * 2 + 1)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Mark the peak
        ax.plot(
            peak_frame,
            probs_array[peak_frame],
            "ro",
            markersize=12,
            label=f"peak (frame {peak_frame}, prob={probs_array[peak_frame]:.4f})",
        )

        # Mark peak boundaries if available
        if "details" in peak and peak["details"].get("left_base") is not None:
            left_base = peak["details"]["left_base"]
            if f_start <= left_base < f_end:
                ax.axvline(
                    x=left_base, color="green", linestyle="--", alpha=0.6, linewidth=1
                )
                ax.plot(
                    left_base,
                    probs_array[left_base],
                    "g^",
                    markersize=8,
                    label=f"left base (frame {left_base})",
                )

        if "details" in peak and peak["details"].get("right_base") is not None:
            right_base = peak["details"]["right_base"]
            if f_start <= right_base < f_end:
                ax.axvline(
                    x=right_base, color="red", linestyle="--", alpha=0.6, linewidth=1
                )
                ax.plot(
                    right_base,
                    probs_array[right_base],
                    "rv",
                    markersize=8,
                    label=f"right base (frame {right_base})",
                )

        # Highlight context window
        context_start_frame = max(0, peak_frame - context_frames)
        context_end_frame = min(n_frames - 1, peak_frame + context_frames)
        ax.axvspan(
            context_start_frame,
            context_end_frame,
            alpha=0.1,
            color="yellow",
            label="audio context",
        )

        ax.set_title(
            f"peak · {idx:03d}  "
            f"[{peak['start_s']:.3f} s]  "
            f"probability: {peak['details'].get('peak_probability', 'N/A') if 'details' in peak else 'N/A':.4f}",
            fontsize=12,
        )
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="upper right")

        plt.tight_layout()
        plot_path = seg_dir / "plot.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()

    console.print(
        f"  [dim]Peak {idx:03d}:[/dim] "
        f"frame: {peak['frame_start']} | "
        f"time: {peak['start_s']:.3f}s | "
        f"prob: {peak['details'].get('peak_probability', 'N/A') if 'details' in peak else 'N/A':.4f} | "
        f"prominence: {peak['details'].get('prominence', 'N/A') if 'details' in peak else 'N/A'}"
    )

# Generate overview plot showing all peaks on the full probability curve
if n_frames > 0:
    fig, ax = plt.subplots(figsize=(16, 6))
    all_frames = np.arange(n_frames)
    ax.plot(
        all_frames, probs_array, "b-", linewidth=1.5, label="VAD Probability", alpha=0.7
    )

    # Mark all peaks
    peak_frames = [p["frame_start"] for p in peaks]
    peak_probs = [probs_array[f] for f in peak_frames]
    ax.scatter(
        peak_frames,
        peak_probs,
        c="red",
        s=80,
        zorder=5,
        label=f"Peaks ({len(peaks)})",
        alpha=0.8,
    )

    # Add peak indices
    for idx, (frame, prob) in enumerate(zip(peak_frames, peak_probs)):
        ax.annotate(
            str(idx),
            (frame, prob),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color="darkred",
        )

    ax.set_title(
        f"All Peaks Overview · {len(peaks)} peaks detected  "
        f"[height={args.height}, distance={args.distance}, prominence={args.prominence}]",
        fontsize=14,
    )
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Speech Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    overview_path = output_dir / "peaks_overview.png"
    plt.savefig(str(overview_path), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Overview plot saved to:[/green] {linkify(overview_path)}")

# Save summary JSONs
summary_path = output_dir / "peaks.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

segs_summary_path = segs_out_dir / "peaks.json"
with open(segs_summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

console.print(f"[bold green]Peaks saved under:[/bold green] {linkify(segs_out_dir)}")
console.print(
    f"[bold green]Peaks summary saved to:[/bold green] {linkify(summary_path)}"
)
