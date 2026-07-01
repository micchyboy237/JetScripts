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
from jet.audio.speech.vad_extractors import extract_troughs, extract_valleys, load_probs
from jet.audio.speech.vad_valley_utils import ThresholdStrategy
from jet.audio.utils.info import display_audio_info
from rich.console import Console

console = Console()
DEFAULT_AUDIO = "/Users/jethroestrada/.cache/files/audio/recording_3_speakers.wav"
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
parser = argparse.ArgumentParser(
    description="Extract VAD valley regions (low-probability silence stretches) from audio"
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
    "--threshold",
    "-t",
    type=float,
    default=None,
    help="Frames below this value are considered silent (default: auto-computed via OTSU).",
)
parser.add_argument(
    "--min-duration",
    "-d",
    type=float,
    default=0.0,
    help="Minimum valley duration in seconds (default: 0.0).",
)
parser.add_argument(
    "--include-troughs",
    action="store_true",
    help="Extract and include trough information within each valley.",
)
parser.add_argument(
    "--trough-height",
    type=float,
    default=None,
    help="Maximum probability for trough detection within valleys.",
)
parser.add_argument(
    "--trough-prominence",
    type=float,
    default=0.15,
    help="Required prominence of troughs within valleys (default: 0.15).",
)
parser.add_argument(
    "--trough-distance",
    type=int,
    default=5,
    help="Minimum frames between troughs within valleys (default: 5).",
)
parser.add_argument(
    "--auto-threshold-strategy",
    type=str,
    default="otsu",
    choices=[s.value for s in ThresholdStrategy],
    help="Strategy for auto-computing threshold if not set (default: otsu).",
)
parser.add_argument(
    "--quantize",
    "-q",
    action="store_true",
    help="Quantize audio to int16 before processing (default: False)",
)
args = parser.parse_args()
min_duration_s = args.min_duration
auto_threshold_strategy = ThresholdStrategy(args.auto_threshold_strategy)
output_dir = Path(args.output_dir)
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)
console.print(
    f"[cyan]threshold={args.threshold}, min_duration_s={min_duration_s:.3f}s[/cyan]"
)
console.print(f"[cyan]auto_threshold_strategy={auto_threshold_strategy.value}[/cyan]")
console.print(f"[cyan]include_troughs={args.include_troughs}[/cyan]")
probs, audio_np = load_probs(args.audio_path)
if args.quantize:
    audio_np = convert_audio_dtype(audio_np, "int16")
display_audio_info(audio_np)
# Extract troughs if requested
troughs = None
if args.include_troughs:
    console.print("[cyan]Extracting troughs for valley analysis...[/cyan]")
    troughs = extract_troughs(
        probs_or_audio=probs,
        frame_shift_ms=FRAME_SHIFT_MS,
        height=args.trough_height,
        distance=args.trough_distance,
        prominence=args.trough_prominence,
        auto_threshold_strategy=auto_threshold_strategy,
    )
    console.print(f"[cyan]Found {len(troughs)} trough(s) for valley analysis[/cyan]")

valleys = extract_valleys(
    probs_or_audio=probs,
    frame_shift_ms=FRAME_SHIFT_MS,
    threshold=args.threshold,
    min_duration_s=min_duration_s,
    troughs=troughs,
    auto_threshold_strategy=auto_threshold_strategy,
)
# Save valleys as individual segment directories with audio, probs, and plots
segs_out_dir = output_dir / "valleys"
segs_out_dir.mkdir(parents=True, exist_ok=True)
all_segments_meta = []
probs_array = np.array(probs, dtype=float) if probs else np.array([])
n_frames = len(probs_array)
total_audio_samples = len(audio_np) if audio_np is not None else 0

for idx, valley in enumerate(valleys):
    seg_dir = segs_out_dir / f"valley_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Save valley metadata
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(valley, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(valley)

    # Save audio slice for this valley
    if audio_np is not None and total_audio_samples > 0:
        start_sample = int(valley["start_s"] * SAMPLE_RATE)
        end_sample = int(valley["end_s"] * SAMPLE_RATE)
        start_sample = max(0, start_sample)
        end_sample = min(total_audio_samples, end_sample)
        audio_slice = audio_np[start_sample:end_sample]
        if len(audio_slice) > 0:
            sf.write(str(seg_dir / "sound.wav"), audio_slice, SAMPLE_RATE)
            console.print(
                f"  [dim]Valley {idx:03d}:[/dim] audio samples [{start_sample}:{end_sample}] "
                f"({len(audio_slice)} samples, {len(audio_slice) / SAMPLE_RATE:.3f}s)"
            )

    # Save valley probability details
    if "details" in valley:
        probs_info = {
            "frame_shift_sec": FRAME_SHIFT_MS / 1000.0,
            "frame_start": valley["frame_start"],
            "frame_end": valley["frame_end"],
            "threshold": valley["details"].get("threshold"),
            "min_probability": valley["details"].get("min_probability"),
            "min_prob_frame": valley["details"].get("min_prob_frame"),
            "min_prob_s": valley["details"].get("min_prob_s"),
            "mean_probability": valley["details"].get("mean_probability"),
            "frame_count": valley["details"].get("frame_count"),
        }

        # Include trough information if available
        if "troughs" in valley["details"] and valley["details"]["troughs"]:
            probs_info["contained_troughs"] = [
                {
                    "frame": t["frame_start"],
                    "probability": t["details"].get("trough_probability"),
                    "prominence": t["details"].get("prominence"),
                }
                for t in valley["details"]["troughs"]
            ]
            probs_info["num_contained_troughs"] = len(valley["details"]["troughs"])

        with open(seg_dir / "probs_info.json", "w", encoding="utf-8") as fh:
            json.dump(probs_info, fh, ensure_ascii=False, indent=2)

    # Generate probability plot for this valley
    if n_frames > 0:
        f_start = max(0, valley["frame_start"] - 10)  # Small context buffer
        f_end = min(n_frames, valley["frame_end"] + 11)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Highlight the valley region
        ax.axvspan(
            valley["frame_start"],
            valley["frame_end"] - 1,
            alpha=0.15,
            color="purple",
            label="valley region",
        )

        # Mark valley boundaries
        ax.axvline(
            x=valley["frame_start"],
            color="purple",
            linestyle="--",
            linewidth=1.5,
            label=f"start (frame {valley['frame_start']})",
        )
        ax.axvline(
            x=valley["frame_end"] - 1,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"end (frame {valley['frame_end'] - 1})",
        )

        # Mark threshold line
        threshold = (
            valley["details"].get("threshold", 0.3) if "details" in valley else 0.3
        )
        ax.axhline(
            y=threshold,
            color="orange",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
            label=f"threshold ({threshold:.3f})",
        )

        # Mark minimum probability point
        if "details" in valley:
            min_frame = valley["details"].get("min_prob_frame")
            if min_frame is not None and f_start <= min_frame < f_end:
                ax.plot(
                    min_frame,
                    valley["details"]["min_probability"],
                    "mo",
                    markersize=10,
                    label=f"min prob (frame {min_frame})",
                )

        # Mark contained troughs
        if "details" in valley and "troughs" in valley["details"]:
            for trough in valley["details"]["troughs"]:
                trough_frame = trough["frame_start"]
                if f_start <= trough_frame < f_end:
                    ax.plot(
                        trough_frame,
                        trough["details"].get("trough_probability", 0),
                        "rv",
                        markersize=8,
                        markerfacecolor="none",
                        label="contained trough"
                        if trough == valley["details"]["troughs"][0]
                        else None,
                    )

        ax.set_title(
            f"valley · {idx:03d}  "
            f"[{valley['start_s']:.3f} s – {valley['end_s']:.3f} s]  "
            f"duration: {valley['duration_s']:.3f}s",
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
        f"  [dim]Valley {idx:03d}:[/dim] "
        f"duration: {valley['duration_s']:.2f}s | "
        f"frames {valley['frame_start']}-{valley['frame_end'] - 1} | "
        f"min_prob: {valley['details'].get('min_probability', 'N/A') if 'details' in valley else 'N/A':.4f} | "
        f"troughs: {len(valley['details'].get('troughs', [])) if 'details' in valley else 0}"
    )

# Generate overview plot showing all valleys on the full probability curve
if n_frames > 0:
    fig, ax = plt.subplots(figsize=(16, 6))
    all_frames = np.arange(n_frames)
    ax.plot(
        all_frames, probs_array, "b-", linewidth=1.5, label="VAD Probability", alpha=0.7
    )

    # Highlight all valley regions
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(valleys))) if valleys else []
    for idx, valley in enumerate(valleys):
        ax.axvspan(
            valley["frame_start"],
            valley["frame_end"] - 1,
            alpha=0.2,
            color=colors[idx] if len(colors) > 0 else "purple",
            label=f"valley {idx}"
            if idx < 10
            else None,  # Label first 10 to avoid clutter
        )

    # Mark minimum probability points
    min_frames = [v["details"]["min_prob_frame"] for v in valleys if "details" in v]
    min_probs = [v["details"]["min_probability"] for v in valleys if "details" in v]
    if min_frames:
        ax.scatter(
            min_frames,
            min_probs,
            c="magenta",
            s=50,
            zorder=5,
            marker="v",
            label="Min probability points",
            alpha=0.7,
        )

    # Show threshold
    threshold_val = args.threshold if args.threshold is not None else "auto"
    display_threshold = (
        args.threshold
        if args.threshold is not None
        else (probs_array.mean() * 0.5 if n_frames > 0 else 0.3)
    )
    ax.axhline(
        y=display_threshold,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label=f"valley threshold ({threshold_val})",
    )

    ax.set_title(
        f"All Valleys Overview · {len(valleys)} valleys detected  "
        f"[threshold={args.threshold}, min_duration={min_duration_s:.3f}s]",
        fontsize=14,
    )
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Speech Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Only show first 10 valley labels + other elements in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="upper right")

    plt.tight_layout()
    overview_path = output_dir / "valleys_overview.png"
    plt.savefig(str(overview_path), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Overview plot saved to:[/green] {linkify(overview_path)}")

# Save summary JSONs
summary_path = output_dir / "valleys.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

segs_summary_path = segs_out_dir / "valleys.json"
with open(segs_summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

console.print(f"[bold green]Valleys saved under:[/bold green] {linkify(segs_out_dir)}")
console.print(
    f"[bold green]Valleys summary saved to:[/bold green] {linkify(summary_path)}"
)
