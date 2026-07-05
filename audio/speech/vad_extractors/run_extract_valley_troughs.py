import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_FRAME_OFFSET,
    DEFAULT_MIN_TROUGH_OFFSET,
    DEFAULT_MIN_VALLEY_DURATION,
    DEFAULT_MIN_VALLEY_FRAMES,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_TROUGH_DISTANCE,
    DEFAULT_TROUGH_HEIGHT,
    DEFAULT_TROUGH_PROMINENCE,
    DEFAULT_VALLEY_THRESHOLD,
)
from jet.audio.audio_waveform.vad.vad_logging import linkify
from jet.audio.helpers.config import FRAME_SHIFT_MS, SAMPLE_RATE
from jet.audio.normalization.dtype_conversion import convert_audio_dtype
from jet.audio.speech.vad_extractors import extract_valley_troughs, load_probs
from jet.audio.speech.vad_valley_utils import ThresholdStrategy
from jet.audio.utils.info import display_audio_info
from rich.console import Console

console = Console()
DEFAULT_AUDIO = "/Users/jethroestrada/.cache/files/audio/recording_3_speakers.wav"
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
parser = argparse.ArgumentParser(
    description="Extract composite valley troughs (salient silence cut points) from audio"
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
    "--smoothing-window",
    type=int,
    default=DEFAULT_SMOOTHING_WINDOW,
    help=f"Smoothing window size for VAD probabilities (default: {DEFAULT_SMOOTHING_WINDOW}, 0 = disabled).",
)
parser.add_argument(
    "--trough-height",
    type=float,
    default=DEFAULT_TROUGH_HEIGHT,
    help=f"Min trough height (default: {DEFAULT_TROUGH_HEIGHT} - None = auto-computed).",
)
parser.add_argument(
    "--trough-prominence",
    type=float,
    default=DEFAULT_TROUGH_PROMINENCE,
    help=f"Trough prominence (default: {DEFAULT_TROUGH_PROMINENCE}).",
)
parser.add_argument(
    "--trough-distance",
    type=int,
    default=DEFAULT_TROUGH_DISTANCE,
    help=f"Min frames between troughs (default: {DEFAULT_TROUGH_DISTANCE}).",
)
parser.add_argument(
    "--valley-threshold",
    type=float,
    default=DEFAULT_VALLEY_THRESHOLD,
    help=f"Valley threshold (default: {DEFAULT_VALLEY_THRESHOLD}, None = auto-computed).",
)
parser.add_argument(
    "--min-valley-duration",
    type=float,
    default=DEFAULT_MIN_VALLEY_DURATION,
    help=f"Minimum valley duration in seconds (default: {DEFAULT_MIN_VALLEY_DURATION}).",
)
parser.add_argument(
    "--min-valley-frames",
    type=int,
    default=DEFAULT_MIN_VALLEY_FRAMES,
    help=f"Minimum valley frames (default: {DEFAULT_MIN_VALLEY_FRAMES}, overrides duration if set).",
)
parser.add_argument(
    "--frame-offset",
    type=int,
    default=DEFAULT_FRAME_OFFSET,
    help=f"Global frame offset for chunked processing (default: {DEFAULT_FRAME_OFFSET}).",
)
parser.add_argument(
    "--min-trough-offset",
    type=float,
    default=DEFAULT_MIN_TROUGH_OFFSET,
    help=f"Min seconds from start for valid trough (default: {DEFAULT_MIN_TROUGH_OFFSET}).",
)
parser.add_argument(
    "--auto-threshold-strategy",
    type=str,
    default="otsu",
    choices=[s.value for s in ThresholdStrategy],
    help="Strategy for auto-computing thresholds if not set (default: otsu).",
)
parser.add_argument(
    "--context-window",
    "-c",
    type=float,
    default=0.5,
    help="Context window in seconds around each valley trough for audio extraction (default: 0.5)",
)
parser.add_argument(
    "--quantize",
    "-q",
    action="store_true",
    help="Quantize audio to int16 before processing (default: False)",
)
parser.add_argument(
    "--sort-by",
    type=str,
    default="score",
    choices=["score", "time", "probability", "duration"],
    help="Sort valley troughs by: score (final_score desc), time (asc), probability (asc), duration (desc) (default: score)",
)
args = parser.parse_args()
context_window_s = args.context_window
auto_threshold_strategy = ThresholdStrategy(args.auto_threshold_strategy)
output_dir = Path(args.output_dir)
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)
console.print(f"[cyan]smoothing_window={args.smoothing_window}[/cyan]")
console.print(
    f"[cyan]trough_height={args.trough_height}, trough_prominence={args.trough_prominence}, trough_distance={args.trough_distance}[/cyan]"
)
console.print(
    f"[cyan]valley_threshold={args.valley_threshold}, min_valley_duration={args.min_valley_duration}s[/cyan]"
)
console.print(
    f"[cyan]frame_offset={args.frame_offset}, min_trough_offset={args.min_trough_offset}s[/cyan]"
)
console.print(f"[cyan]auto_threshold_strategy={auto_threshold_strategy.value}[/cyan]")
console.print(
    f"[cyan]context_window_s={context_window_s:.3f}s, sort_by={args.sort_by}[/cyan]"
)
probs, audio_np = load_probs(args.audio_path)
if args.quantize:
    audio_np = convert_audio_dtype(audio_np, "int16")
display_audio_info(audio_np)
valley_troughs = extract_valley_troughs(
    probs_or_audio=probs,
    sample_rate=SAMPLE_RATE,
    frame_shift_ms=FRAME_SHIFT_MS,
    smoothing_window=args.smoothing_window,
    trough_height=args.trough_height,
    trough_prominence=args.trough_prominence,
    trough_distance=args.trough_distance,
    valley_threshold=args.valley_threshold,
    min_valley_duration_s=args.min_valley_duration,
    min_valley_frames=args.min_valley_frames,
    frame_offset=args.frame_offset,
    min_trough_offset_s=args.min_trough_offset,
)
# Sort valley troughs based on user preference
if args.sort_by == "score":
    valley_troughs.sort(
        key=lambda vt: vt["valley"].get("final_score", 0.0), reverse=True
    )
elif args.sort_by == "time":
    valley_troughs.sort(key=lambda vt: vt["time_s"])
elif args.sort_by == "probability":
    valley_troughs.sort(key=lambda vt: vt["prob"])
elif args.sort_by == "duration":
    valley_troughs.sort(
        key=lambda vt: vt["valley"].get("duration_s", 0.0), reverse=True
    )

console.print(
    f"[green]Sorted {len(valley_troughs)} valley trough(s) by {args.sort_by}[/green]"
)
# Save valley troughs as individual segment directories with audio, probs, and plots
segs_out_dir = output_dir / "valley_troughs"
segs_out_dir.mkdir(parents=True, exist_ok=True)
all_segments_meta = []
probs_array = np.array(probs, dtype=float) if probs else np.array([])
n_frames = len(probs_array)
total_audio_samples = len(audio_np) if audio_np is not None else 0
context_frames = int(context_window_s / (FRAME_SHIFT_MS / 1000.0))
# Track top 5 for summary
top_5 = valley_troughs[:5] if len(valley_troughs) >= 5 else valley_troughs

for idx, vt in enumerate(valley_troughs):
    valley = vt["valley"]
    seg_dir = segs_out_dir / f"valley_trough_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Build comprehensive metadata
    meta = {
        "rank": idx + 1,
        "sort_by": args.sort_by,
        "trough": {
            "frame": vt["frame"],
            "global_frame": vt["global_frame"],
            "probability": vt["prob"],
            "time_s": vt["time_s"],
            "global_time_s": vt["global_time_s"],
            "prominence": vt["prominence"],
            "width": vt["width"],
        },
        "valley": valley,
    }

    # Save metadata
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(meta)

    # Extract audio context around the valley trough
    if audio_np is not None and total_audio_samples > 0:
        # Extract the full valley audio plus context
        valley_start_s = valley["start_s"]
        valley_end_s = valley["end_s"]
        context_start_s = max(0, valley_start_s - context_window_s)
        context_end_s = min(
            total_audio_samples / SAMPLE_RATE, valley_end_s + context_window_s
        )

        start_sample = int(context_start_s * SAMPLE_RATE)
        end_sample = int(context_end_s * SAMPLE_RATE)
        audio_slice = audio_np[start_sample:end_sample]

        if len(audio_slice) > 0:
            sf.write(str(seg_dir / "sound.wav"), audio_slice, SAMPLE_RATE)

            # Also extract just the trough point audio (small window)
            trough_time = vt["time_s"]
            trough_start_s = max(0, trough_time - 0.05)
            trough_end_s = min(total_audio_samples / SAMPLE_RATE, trough_time + 0.05)
            trough_start_sample = int(trough_start_s * SAMPLE_RATE)
            trough_end_sample = int(trough_end_s * SAMPLE_RATE)
            trough_audio = audio_np[trough_start_sample:trough_end_sample]

            if len(trough_audio) > 0:
                sf.write(str(seg_dir / "trough_point.wav"), trough_audio, SAMPLE_RATE)

            console.print(
                f"  [dim]VT {idx:03d}:[/dim] valley audio [{context_start_s:.3f}s - {context_end_s:.3f}s] "
                f"({len(audio_slice)} samples, {len(audio_slice) / SAMPLE_RATE:.3f}s)"
            )

    # Save detailed probability information
    probs_info = {
        "frame_shift_sec": FRAME_SHIFT_MS / 1000.0,
        "global_frame_offset": args.frame_offset,
        "trough": {
            "local_frame": vt["frame"],
            "global_frame": vt["global_frame"],
            "probability": vt["prob"],
            "time_s": vt["time_s"],
            "global_time_s": vt["global_time_s"],
        },
        "valley": {
            "frame_start": valley["frame_start"],
            "frame_end": valley["frame_end"],
            "frame_length": valley["frame_length"],
            "start_s": valley["start_s"],
            "end_s": valley["end_s"],
            "duration_s": valley["duration_s"],
            "global_frame_start": valley["global_frame_start"],
            "global_frame_end": valley["global_frame_end"],
            "global_start_s": valley["global_start_s"],
            "global_end_s": valley["global_end_s"],
            "global_duration_s": valley["global_duration_s"],
            "is_last": valley["is_last"],
        },
        "scores": {
            "valley_score": valley["valley_score"],
            "trough_score": valley["trough_score"],
            "final_score": valley["final_score"],
        },
    }

    with open(seg_dir / "probs_info.json", "w", encoding="utf-8") as fh:
        json.dump(probs_info, fh, ensure_ascii=False, indent=2)

    # Extract valley probability values
    if n_frames > 0:
        v_start = max(0, valley["frame_start"])
        v_end = min(n_frames, valley["frame_end"])
        valley_probs = probs_array[v_start:v_end].tolist()
        with open(seg_dir / "probs.json", "w", encoding="utf-8") as fh:
            json.dump(valley_probs, fh, ensure_ascii=False, indent=2)

    # Generate probability plot for this valley trough
    if n_frames > 0:
        # Show the valley with generous context
        f_start = max(0, valley["frame_start"] - context_frames)
        f_end = min(n_frames, valley["frame_end"] + context_frames)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Highlight the valley region
        ax.axvspan(
            valley["frame_start"],
            valley["frame_end"] - 1,
            alpha=0.15,
            color="purple",
            label=f"valley [{valley['start_s']:.3f}s - {valley['end_s']:.3f}s]",
        )

        # Mark valley boundaries
        ax.axvline(
            x=valley["frame_start"],
            color="purple",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"valley start (frame {valley['frame_start']})",
        )
        ax.axvline(
            x=valley["frame_end"] - 1,
            color="purple",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"valley end (frame {valley['frame_end'] - 1})",
        )

        # Mark the trough point
        trough_frame = vt["frame"]
        if f_start <= trough_frame < f_end:
            ax.plot(
                trough_frame,
                vt["prob"],
                "ro",
                markersize=12,
                markeredgewidth=2,
                markeredgecolor="darkred",
                label=f"trough (frame {trough_frame}, prob={vt['prob']:.4f})",
            )

            # Add score annotation
            ax.annotate(
                f"Score: {valley['final_score']:.3f}\n"
                f"Valley: {valley['valley_score']:.3f}\n"
                f"Trough: {valley['trough_score']:.3f}",
                xy=(trough_frame, vt["prob"]),
                xytext=(20, 30),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
            )

        # Show probability statistics in the plot
        if valley["frame_length"] > 0:
            valley_probs_slice = probs_array[
                valley["frame_start"] : valley["frame_end"]
            ]
            ax.axhline(
                y=np.mean(valley_probs_slice),
                color="orange",
                linestyle=":",
                linewidth=1,
                alpha=0.6,
                label=f"valley mean prob ({np.mean(valley_probs_slice):.3f})",
            )

        # Add ranking badge for top 5
        is_top = idx < 5
        rank_text = f"#{idx + 1}" if is_top else ""

        ax.set_title(
            f"{'★ ' if is_top else ''}Valley Trough {idx:03d} {rank_text} "
            f"[{valley['start_s']:.3f}s - {valley['end_s']:.3f}s]  "
            f"final_score: {valley['final_score']:.3f}",
            fontsize=12,
            color="darkred" if is_top else "black",
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

    # Log segment info
    console.print(
        f"  [{'bold yellow' if is_top else 'dim'}]VT {idx:03d}:[/{'bold yellow' if is_top else 'dim'}] "
        f"score: {valley['final_score']:.4f} | "
        f"valley_score: {valley['valley_score']:.4f} | "
        f"trough_score: {valley['trough_score']:.4f} | "
        f"prob: {vt['prob']:.4f} | "
        f"time: {vt['time_s']:.3f}s | "
        f"valley_dur: {valley['duration_s']:.3f}s | "
        f"is_last: {valley['is_last']}"
    )

# Generate overview plot showing all valley troughs on the full probability curve
if n_frames > 0:
    fig, ax = plt.subplots(figsize=(18, 8))
    all_frames = np.arange(n_frames)
    ax.plot(
        all_frames, probs_array, "b-", linewidth=1.5, label="VAD Probability", alpha=0.7
    )

    # Highlight all valley regions
    colors = (
        plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(valley_troughs)))
        if valley_troughs
        else []
    )
    for idx, vt in enumerate(valley_troughs):
        valley = vt["valley"]
        color = colors[idx] if len(colors) > 0 else "purple"
        ax.axvspan(
            valley["frame_start"],
            valley["frame_end"] - 1,
            alpha=0.15,
            color=color,
            label=f"valley {idx}" if idx < 5 else None,  # Label top 5 to avoid clutter
        )

    # Mark all trough points with size proportional to score
    if valley_troughs:
        trough_frames = [vt["frame"] for vt in valley_troughs]
        trough_probs = [vt["prob"] for vt in valley_troughs]
        scores = [vt["valley"]["final_score"] for vt in valley_troughs]

        # Normalize scores for marker sizing
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            sizes = [
                30 + 170 * (s - min_score) / (max_score - min_score) for s in scores
            ]
        else:
            sizes = [100] * len(scores)

        scatter = ax.scatter(
            trough_frames,
            trough_probs,
            c=scores,
            s=sizes,
            cmap="RdYlGn",
            edgecolors="black",
            linewidth=1,
            zorder=5,
            alpha=0.8,
            label=f"Valley Troughs ({len(valley_troughs)})",
        )
        cbar = plt.colorbar(scatter, ax=ax, label="Final Score")

        # Add rank labels for top 10
        for idx, (frame, prob, score) in enumerate(
            zip(trough_frames, trough_probs, scores)
        ):
            if idx < 10:
                ax.annotate(
                    f"#{idx + 1}",
                    (frame, prob),
                    xytext=(0, 15),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    color="darkred",
                )

    # Mark last valley if present
    last_valleys = [vt for vt in valley_troughs if vt["valley"]["is_last"]]
    if last_valleys:
        last_vt = last_valleys[0]
        ax.axvline(
            x=last_vt["valley"]["frame_end"] - 1,
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label=f"last valley end (frame {last_vt['valley']['frame_end'] - 1})",
        )

    ax.set_title(
        f"Valley Troughs Overview · {len(valley_troughs)} detected  "
        f"[sorted by {args.sort_by}]  "
        f"smoothing={args.smoothing_window}, valley_dur≥{args.min_valley_duration}s",
        fontsize=14,
    )
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Speech Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="upper right")

    plt.tight_layout()
    overview_path = output_dir / "valley_troughs_overview.png"
    plt.savefig(str(overview_path), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Overview plot saved to:[/green] {linkify(overview_path)}")

# Generate top 5 detailed comparison plot
if len(valley_troughs) >= 2:
    fig, axes = plt.subplots(
        min(5, len(valley_troughs)), 1, figsize=(14, 3 * min(5, len(valley_troughs)))
    )
    if len(valley_troughs) == 1:
        axes = [axes]

    for idx, (ax_curr, vt) in enumerate(zip(axes, valley_troughs[:5])):
        valley = vt["valley"]
        f_start = max(0, valley["frame_start"] - 20)
        f_end = min(n_frames, valley["frame_end"] + 20)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]

        ax_curr.plot(frames, zoomed, "b-", linewidth=2, alpha=0.8)
        ax_curr.axvspan(
            valley["frame_start"], valley["frame_end"] - 1, alpha=0.2, color="purple"
        )
        ax_curr.plot(vt["frame"], vt["prob"], "ro", markersize=10)

        ax_curr.set_title(
            f"#{idx + 1}: Score={valley['final_score']:.3f} | "
            f"Valley=[{valley['start_s']:.3f}s-{valley['end_s']:.3f}s] | "
            f"Trough={vt['time_s']:.3f}s (prob={vt['prob']:.3f})",
            fontsize=10,
        )
        ax_curr.set_ylim(-0.05, 1.05)
        ax_curr.grid(True, alpha=0.3)

    plt.tight_layout()
    top5_path = output_dir / "top_5_valley_troughs.png"
    plt.savefig(str(top5_path), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(
        f"[green]Top 5 comparison plot saved to:[/green] {linkify(top5_path)}"
    )

# Generate score distribution plot
if len(valley_troughs) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Score histogram
    scores = [vt["valley"]["final_score"] for vt in valley_troughs]
    axes[0].hist(scores, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(
        np.mean(scores),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(scores):.3f}",
    )
    axes[0].axvline(
        np.median(scores),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(scores):.3f}",
    )
    axes[0].set_title("Final Score Distribution")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Valley duration vs score
    durations = [vt["valley"]["duration_s"] for vt in valley_troughs]
    probs_vt = [vt["prob"] for vt in valley_troughs]
    scatter = axes[1].scatter(
        durations, scores, c=probs_vt, cmap="coolwarm", s=80, alpha=0.7
    )
    axes[1].set_title("Valley Duration vs Final Score")
    axes[1].set_xlabel("Valley Duration (s)")
    axes[1].set_ylabel("Final Score")
    plt.colorbar(scatter, ax=axes[1], label="Trough Probability")
    axes[1].grid(True, alpha=0.3)

    # Time vs score
    times = [vt["time_s"] for vt in valley_troughs]
    axes[2].scatter(times, scores, c="purple", s=80, alpha=0.7)
    axes[2].set_title("Time Position vs Final Score")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Final Score")
    axes[2].grid(True, alpha=0.3)

    # Add trend line
    if len(times) > 1:
        z = np.polyfit(times, scores, 1)
        p = np.poly1d(z)
        axes[2].plot(times, p(times), "r--", alpha=0.5, label="Trend")
        axes[2].legend()

    plt.tight_layout()
    distribution_path = output_dir / "score_distribution.png"
    plt.savefig(str(distribution_path), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(
        f"[green]Score distribution plots saved to:[/green] {linkify(distribution_path)}"
    )

# Save comprehensive summary JSON
summary = {
    "total_valley_troughs": len(valley_troughs),
    "parameters": {
        "smoothing_window": args.smoothing_window,
        "trough_height": args.trough_height,
        "trough_prominence": args.trough_prominence,
        "trough_distance": args.trough_distance,
        "valley_threshold": args.valley_threshold,
        "min_valley_duration_s": args.min_valley_duration,
        "min_valley_frames": args.min_valley_frames,
        "frame_offset": args.frame_offset,
        "min_trough_offset_s": args.min_trough_offset,
        "auto_threshold_strategy": auto_threshold_strategy.value,
        "sort_by": args.sort_by,
    },
    "statistics": {
        "mean_score": float(
            np.mean([vt["valley"]["final_score"] for vt in valley_troughs])
        )
        if valley_troughs
        else 0,
        "median_score": float(
            np.median([vt["valley"]["final_score"] for vt in valley_troughs])
        )
        if valley_troughs
        else 0,
        "std_score": float(
            np.std([vt["valley"]["final_score"] for vt in valley_troughs])
        )
        if valley_troughs
        else 0,
        "max_score": float(max([vt["valley"]["final_score"] for vt in valley_troughs]))
        if valley_troughs
        else 0,
        "min_score": float(min([vt["valley"]["final_score"] for vt in valley_troughs]))
        if valley_troughs
        else 0,
        "mean_valley_duration": float(
            np.mean([vt["valley"]["duration_s"] for vt in valley_troughs])
        )
        if valley_troughs
        else 0,
        "mean_trough_probability": float(np.mean([vt["prob"] for vt in valley_troughs]))
        if valley_troughs
        else 0,
        "last_valley_count": sum(1 for vt in valley_troughs if vt["valley"]["is_last"]),
    },
    "top_5": [
        {
            "rank": i + 1,
            "final_score": vt["valley"]["final_score"],
            "valley_score": vt["valley"]["valley_score"],
            "trough_score": vt["valley"]["trough_score"],
            "trough_probability": vt["prob"],
            "trough_time_s": vt["time_s"],
            "valley_duration_s": vt["valley"]["duration_s"],
            "valley_start_s": vt["valley"]["start_s"],
            "valley_end_s": vt["valley"]["end_s"],
            "is_last": vt["valley"]["is_last"],
        }
        for i, vt in enumerate(top_5)
    ],
    "valley_troughs": all_segments_meta,
}

summary_path = output_dir / "valley_troughs.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)

segs_summary_path = segs_out_dir / "valley_troughs.json"
with open(segs_summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

# Print top 5 summary table
console.print("\n[bold cyan]═══ Top 5 Valley Troughs ═══[/bold cyan]")
console.print(
    f"{'Rank':<6} {'Score':<10} {'Valley':<10} {'Trough':<10} {'Prob':<10} {'Time':<10} {'Duration':<10} {'Prom':<10} {'Width':<10}"
)
console.print("-" * 82)
for i, vt in enumerate(top_5):
    console.print(
        f"#{i + 1:<5} "
        f"{vt['valley']['final_score']:<10.4f} "
        f"{vt['valley']['valley_score']:<10.4f} "
        f"{vt['valley']['trough_score']:<10.4f} "
        f"{vt['prob']:<10.4f} "
        f"{vt['time_s']:<10.3f} "
        f"{vt['valley']['duration_s']:<10.3f} "
        f"{vt['prominence']:<10.4f} "
        f"{vt['width']:<10.2f}"
    )

console.print(
    f"\n[bold green]Valley troughs saved under:[/bold green] {linkify(segs_out_dir)}"
)
console.print(
    f"[bold green]Valley troughs summary saved to:[/bold green] {linkify(summary_path)}"
)
