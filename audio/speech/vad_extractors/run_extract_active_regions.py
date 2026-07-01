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
from jet.audio.speech.vad_extractors import extract_active_regions, load_probs
from jet.audio.utils.info import display_audio_info
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
    default=0.0,
    help="Minimum segment duration in seconds (default: 0.0)",
)
parser.add_argument(
    "--quantize",
    "-q",
    action="store_true",
    help="Quantize audio to int16 before processing (default: False)",
)
args = parser.parse_args()
frame_offset = 0
smoothing_window = 0
min_duration_s = args.min_duration
output_dir = Path(args.output_dir)
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)
console.print(f"[cyan]min_duration_s={min_duration_s:.3f}s[/cyan]")
probs, audio_np = load_probs(args.audio_path)
if args.quantize:
    audio_np = convert_audio_dtype(audio_np, "int16")
display_audio_info(audio_np)
active_regions = extract_active_regions(
    probs_or_audio=audio_np,
    frame_shift_ms=FRAME_SHIFT_MS,
    min_duration_s=min_duration_s,
)
# Save active regions as individual segment directories with audio, probs, and plots
segs_out_dir = output_dir / "active_regions"
segs_out_dir.mkdir(parents=True, exist_ok=True)
all_segments_meta = []
probs_array = np.array(probs, dtype=float) if probs else np.array([])
n_frames = len(probs_array)
total_audio_samples = len(audio_np) if audio_np is not None else 0

for idx, region in enumerate(active_regions):
    seg_dir = segs_out_dir / f"segment_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Save segment metadata
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(region, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(region)

    # Save audio slice for this active region
    if audio_np is not None and total_audio_samples > 0:
        start_sample = int(region["start_s"] * SAMPLE_RATE)
        end_sample = int(region["end_s"] * SAMPLE_RATE)
        start_sample = max(0, start_sample)
        end_sample = min(total_audio_samples, end_sample)
        audio_slice = audio_np[start_sample:end_sample]
        if len(audio_slice) > 0:
            sf.write(str(seg_dir / "sound.wav"), audio_slice, SAMPLE_RATE)
            console.print(
                f"  [dim]Segment {idx:03d}:[/dim] audio samples [{start_sample}:{end_sample}] "
                f"({len(audio_slice)} samples, {len(audio_slice) / SAMPLE_RATE:.3f}s)"
            )

    # Save region probabilities
    if "details" in region and "region_probs" in region["details"]:
        with open(seg_dir / "probs.json", "w", encoding="utf-8") as fh:
            json.dump(
                region["details"]["region_probs"], fh, ensure_ascii=False, indent=2
            )

    # Save probability statistics
    if "details" in region:
        probs_info = {
            "frame_shift_sec": FRAME_SHIFT_MS / 1000.0,
            "frame_start": region["frame_start"],
            "frame_end": region["frame_end"],
            "threshold": region["details"].get("threshold"),
            "max_probability": region["details"].get("max_probability"),
            "mean_probability": region["details"].get("mean_probability"),
            "frame_count": region["details"].get("frame_count"),
        }
        with open(seg_dir / "probs_info.json", "w", encoding="utf-8") as fh:
            json.dump(probs_info, fh, ensure_ascii=False, indent=2)

    # Generate probability plot for this segment
    if n_frames > 0:
        f_start = max(0, region["frame_start"])
        f_end = min(n_frames, region["frame_end"] + 1)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Highlight the active region
        ax.axvspan(f_start, f_end - 1, alpha=0.15, color="green", label="active region")

        # Mark region boundaries
        ax.axvline(
            x=region["frame_start"],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"start (frame {region['frame_start']})",
        )
        ax.axvline(
            x=region["frame_end"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"end (frame {region['frame_end']})",
        )

        # Mark threshold line
        threshold = (
            region["details"].get("threshold", 0.3) if "details" in region else 0.3
        )
        ax.axhline(
            y=threshold,
            color="orange",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
            label=f"threshold ({threshold})",
        )

        ax.set_title(
            f"active_region · segment {idx:03d}  "
            f"[{region['start_s']:.3f} s – {region['end_s']:.3f} s]",
            fontsize=12,
        )
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Speech Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="upper right")

        plt.tight_layout()
        plot_path = seg_dir / "plot.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()

    console.print(
        f"  [dim]Segment {idx:03d}:[/dim] "
        f"duration: {region['duration_s']:.2f}s | "
        f"frames {region['frame_start']}-{region['frame_end']} | "
        f"max_prob: {region['details'].get('max_probability', 'N/A') if 'details' in region else 'N/A'}"
    )

# Save summary JSON (same as before but in the new location)
summary_path = output_dir / "active_regions.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

# Also save a copy in the segments directory for consistency
active_regions_output = segs_out_dir / "active_regions.json"
with open(active_regions_output, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

console.print(
    f"[bold green]Active regions saved to:[/bold green] {linkify(segs_out_dir)}"
)
console.print(
    f"[bold green]Active regions summary saved to:[/bold green] {linkify(summary_path)}"
)
