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
from jet.audio.speech.vad_extractors import extract_trough_to_trough, load_probs
from jet.audio.utils.info import display_audio_info
from rich.console import Console
from rich.table import Table

console = Console()
DEFAULT_AUDIO = "/Users/jethroestrada/.cache/files/audio/recording_3_speakers.wav"
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
parser = argparse.ArgumentParser(
    description="Segment audio from trough to trough using VAD speech probabilities"
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
    help="Minimum segment duration in seconds (default: 0.0).",
)
parser.add_argument(
    "--max-duration",
    type=float,
    default=None,
    help="Maximum segment duration in seconds (default: no limit).",
)
parser.add_argument(
    "--context-window",
    "-c",
    type=float,
    default=0.3,
    help="Extra context window in seconds around segments for audio extraction (default: 0.3).",
)
parser.add_argument(
    "--sort-by",
    type=str,
    default="time",
    choices=["time", "duration", "mean_prob", "max_prob", "min_prob"],
    help="Sort segments by: time (asc), duration (desc), mean_prob (desc), max_prob (desc), min_prob (desc) (default: time)",
)
parser.add_argument(
    "--quantize",
    "-q",
    action="store_true",
    help="Quantize audio to int16 before processing (default: False)",
)
parser.add_argument(
    "--save-concatenated",
    action="store_true",
    help="Save a concatenated audio file of all segments with silence markers (default: False)",
)
parser.add_argument(
    "--silence-marker-duration",
    type=float,
    default=0.1,
    help="Duration of silence marker between segments in concatenated output (default: 0.1s)",
)

# extract_valley_troughs params
parser.add_argument(
    "--smoothing-window",
    type=int,
    default=0,
    help="Smoothing window size for VAD probabilities (0 = disabled).",
)
parser.add_argument(
    "--trough-height",
    type=float,
    default=None,
    help="Min trough height (None = auto-computed).",
)
parser.add_argument(
    "--trough-prominence",
    type=float,
    default=0.15,
    help="Trough prominence (default: 0.15).",
)
parser.add_argument(
    "--trough-distance",
    type=int,
    default=5,
    help="Min frames between troughs (default: 5).",
)
parser.add_argument(
    "--valley-threshold",
    type=float,
    default=None,
    help="Valley threshold (None = auto-computed).",
)
parser.add_argument(
    "--min-valley-duration",
    type=float,
    default=0.25,
    help="Minimum valley duration in seconds (default: 0.25).",
)
parser.add_argument(
    "--min-valley-frames",
    type=int,
    default=None,
    help="Minimum valley frames (overrides duration if set).",
)
parser.add_argument(
    "--frame-offset",
    type=int,
    default=0,
    help="Global frame offset for chunked processing.",
)
parser.add_argument(
    "--min-trough-offset",
    type=float,
    default=0.4,
    help="Min seconds from start for valid trough (default: 0.4).",
)
args = parser.parse_args()
frame_offset = 0
smoothing_window = 0
min_duration_s = args.min_duration
max_duration_s = args.max_duration
context_window_s = args.context_window
output_dir = Path(args.output_dir)
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)
console.print(
    f"[cyan]min_duration_s={min_duration_s:.3f}s, max_duration_s={max_duration_s}[/cyan]"
)
console.print(
    f"[cyan]context_window_s={context_window_s:.3f}s, sort_by={args.sort_by}[/cyan]"
)
_, audio_np = load_probs(args.audio_path)
if args.quantize:
    audio_np = convert_audio_dtype(audio_np, "int16")
display_audio_info(audio_np)
segments_with_audio, probs = extract_trough_to_trough(
    probs_or_audio=audio_np,
    frame_shift_ms=FRAME_SHIFT_MS,
    sample_rate=SAMPLE_RATE,
    with_audio=True,
    with_scores=True,
    min_duration_s=min_duration_s,
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
# Apply maximum duration filter if specified
if max_duration_s is not None and max_duration_s > 0:
    before_filter = len(segments_with_audio)
    segments_with_audio = [
        (seg, aud)
        for seg, aud in segments_with_audio
        if seg["duration_s"] <= max_duration_s
    ]
    filtered_count = before_filter - len(segments_with_audio)
    if filtered_count > 0:
        console.print(
            f"[yellow]Filtered {filtered_count} segment(s) longer than "
            f"{max_duration_s:.3f}s. Kept {len(segments_with_audio)} segment(s).[/yellow]"
        )

# Sort segments based on user preference
if args.sort_by == "duration":
    segments_with_audio.sort(key=lambda x: x[0]["duration_s"], reverse=True)
elif args.sort_by == "mean_prob":
    segments_with_audio.sort(
        key=lambda x: x[0].get("prob_stats", {}).get("mean", 0), reverse=True
    )
elif args.sort_by == "max_prob":
    segments_with_audio.sort(
        key=lambda x: x[0].get("prob_stats", {}).get("max", 0), reverse=True
    )
elif args.sort_by == "min_prob":
    segments_with_audio.sort(
        key=lambda x: x[0].get("prob_stats", {}).get("min", 0), reverse=True
    )
# Default: keep time order

console.print(
    f"[green]Sorted {len(segments_with_audio)} segment(s) by {args.sort_by}[/green]"
)
# Save segments as individual directories with audio, probs, and enhanced plots
segs_out_dir = output_dir / "trough_to_trough"
segs_out_dir.mkdir(parents=True, exist_ok=True)
all_segments_meta = []
probs_array = np.array(probs, dtype=float) if probs else np.array([])
n_frames = len(probs_array)
total_audio_samples = len(audio_np) if audio_np is not None else 0
context_samples = int(context_window_s * SAMPLE_RATE)
# Track statistics
total_duration = 0.0
segment_durations = []
segment_mean_probs = []

for idx, (seg_meta, audio_slice) in enumerate(segments_with_audio):
    seg_dir = segs_out_dir / f"segment_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Enhanced metadata with sorting rank
    enhanced_meta = {
        **seg_meta,
        "rank": idx + 1,
        "sort_by": args.sort_by,
        "original_index": idx,  # Will be updated after sorting
    }

    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(enhanced_meta, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(enhanced_meta)

    # Track statistics
    total_duration += seg_meta["duration_s"]
    segment_durations.append(seg_meta["duration_s"])
    if seg_meta.get("prob_stats"):
        segment_mean_probs.append(seg_meta["prob_stats"]["mean"])

    # Extract audio with context window
    if total_audio_samples > 0:
        # Core segment audio
        start_sample = int(seg_meta["start_s"] * SAMPLE_RATE)
        end_sample = int(seg_meta["end_s"] * SAMPLE_RATE)
        start_sample = max(0, start_sample)
        end_sample = min(total_audio_samples, end_sample)
        core_audio = audio_np[start_sample:end_sample]

        if len(core_audio) > 0:
            sf.write(str(seg_dir / "sound.wav"), core_audio, SAMPLE_RATE)

        # Context-extended audio
        context_start_sample = max(0, start_sample - context_samples)
        context_end_sample = min(total_audio_samples, end_sample + context_samples)
        context_audio = audio_np[context_start_sample:context_end_sample]

        if len(context_audio) > 0:
            sf.write(
                str(seg_dir / "sound_with_context.wav"), context_audio, SAMPLE_RATE
            )
            console.print(
                f"  [dim]Segment {idx:03d}:[/dim] "
                f"core=[{start_sample}:{end_sample}] ({len(core_audio)} samples) | "
                f"context=[{context_start_sample}:{context_end_sample}] "
                f"({len(context_audio)} samples)"
            )

    # Save segment probabilities
    if seg_meta.get("segment_probs"):
        with open(seg_dir / "probs.json", "w", encoding="utf-8") as fh:
            json.dump(seg_meta["segment_probs"], fh, ensure_ascii=False, indent=2)

        if seg_meta.get("prob_stats"):
            probs_info = {
                "frame_shift_sec": FRAME_SHIFT_MS / 1000.0,
                "frame_start": seg_meta["start_frame"],
                "frame_end": seg_meta["end_frame"],
                "stats": seg_meta["prob_stats"],
                "trough_start": seg_meta.get("trough_start"),
                "trough_end": seg_meta.get("trough_end"),
                "is_boundary_segment": seg_meta.get("trough_start") is None
                or seg_meta.get("trough_end") is None,
            }
            with open(seg_dir / "probs_info.json", "w", encoding="utf-8") as fh:
                json.dump(probs_info, fh, ensure_ascii=False, indent=2)

    # Generate enhanced probability plot for this segment
    if n_frames > 0:
        f_start = max(0, seg_meta["start_frame"] - 20)  # Extra context frames
        f_end = min(n_frames, seg_meta["end_frame"] + 21)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Highlight the trough-to-trough span
        ax.axvspan(
            seg_meta["start_frame"],
            seg_meta["end_frame"],
            alpha=0.12,
            color="purple",
            label="trough span",
        )

        # Mark boundaries
        is_first = seg_meta.get("trough_start") is None
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

        is_last = seg_meta.get("trough_end") is None
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

        # Add probability statistics overlay
        if seg_meta.get("prob_stats"):
            stats = seg_meta["prob_stats"]
            mean_prob = stats["mean"]
            ax.axhline(
                y=mean_prob,
                color="green",
                linestyle=":",
                linewidth=1,
                alpha=0.6,
                label=f"mean prob ({mean_prob:.3f})",
            )

            # Add stats text box
            stats_text = (
                f"Mean: {stats['mean']:.3f}\n"
                f"Max: {stats['max']:.3f}\n"
                f"Min: {stats['min']:.3f}\n"
                f"Std: {stats['std']:.3f}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                fontsize=8,
            )

        # Highlight speech segments within this trough span
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

        # Add ranking badge for top 5
        is_top = idx < 5
        rank_text = f"#{idx + 1}" if is_top and args.sort_by != "time" else ""

        ax.set_title(
            f"{'★ ' if is_top and args.sort_by != 'time' else ''}"
            f"trough_to_trough · segment {idx:03d} {rank_text} "
            f"[{seg_meta['start_s']:.3f}s – {seg_meta['end_s']:.3f}s]  "
            f"duration: {seg_meta['duration_s']:.3f}s",
            fontsize=12,
            color="darkred" if is_top and args.sort_by != "time" else "black",
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

    # Enhanced console output
    prob_stats_str = ""
    if seg_meta.get("prob_stats"):
        ps = seg_meta["prob_stats"]
        prob_stats_str = f" | mean_prob: {ps['mean']:.3f} | max_prob: {ps['max']:.3f}"

    is_boundary = (
        seg_meta.get("trough_start") is None or seg_meta.get("trough_end") is None
    )
    boundary_marker = " [BOUNDARY]" if is_boundary else ""

    console.print(
        f"  [{'bold yellow' if is_top and args.sort_by != 'time' else 'dim'}]"
        f"Segment {idx:03d}:[/{'bold yellow' if is_top and args.sort_by != 'time' else 'dim'}] "
        f"duration: {seg_meta['duration_s']:.2f}s | "
        f"frames {seg_meta['start_frame']}-{seg_meta['end_frame']} | "
        f"{len(zoomed)} frames{boundary_marker}"
        f"{prob_stats_str}"
    )

# Generate overview plot showing all segments
if n_frames > 0 and segments_with_audio:
    fig, ax = plt.subplots(figsize=(18, 8))
    all_frames = np.arange(n_frames)
    ax.plot(
        all_frames, probs_array, "b-", linewidth=1.5, label="VAD Probability", alpha=0.7
    )

    # Highlight all segments with alternating colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(segments_with_audio)))
    for idx, (seg_meta, _) in enumerate(segments_with_audio):
        ax.axvspan(
            seg_meta["start_frame"],
            seg_meta["end_frame"],
            alpha=0.15,
            color=colors[idx] if len(colors) > 0 else "purple",
            label=f"seg {idx}" if idx < 5 else None,  # Label first 5 to avoid clutter
        )

        # Mark trough boundaries
        if seg_meta.get("trough_start"):
            ax.axvline(
                x=seg_meta["start_frame"],
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )
        if seg_meta.get("trough_end"):
            ax.axvline(
                x=seg_meta["end_frame"],
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )

    # Mark trough points
    trough_frames = []
    trough_probs = []
    for seg_meta, _ in segments_with_audio:
        if seg_meta.get("trough_start") and seg_meta["start_frame"] < n_frames:
            trough_frames.append(seg_meta["start_frame"])
            trough_probs.append(probs_array[seg_meta["start_frame"]])
        if seg_meta.get("trough_end") and seg_meta["end_frame"] < n_frames:
            trough_frames.append(seg_meta["end_frame"])
            trough_probs.append(probs_array[seg_meta["end_frame"]])

    if trough_frames:
        ax.scatter(
            trough_frames,
            trough_probs,
            c="red",
            s=50,
            zorder=5,
            marker="v",
            alpha=0.7,
            label=f"Troughs ({len(trough_frames)})",
        )

    ax.set_title(
        f"Trough-to-Trough Segmentation Overview · {len(segments_with_audio)} segments  "
        f"[min_duration={min_duration_s:.3f}s]  [sorted by {args.sort_by}]",
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
    overview_path = output_dir / "segments_overview.png"
    plt.savefig(str(overview_path), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Overview plot saved to:[/green] {linkify(overview_path)}")

# Generate segment duration distribution plot
if segment_durations:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Duration histogram
    axes[0].hist(
        segment_durations,
        bins=min(30, len(segment_durations)),
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].axvline(
        np.mean(segment_durations),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(segment_durations):.3f}s",
    )
    axes[0].axvline(
        np.median(segment_durations),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(segment_durations):.3f}s",
    )
    axes[0].set_title("Segment Duration Distribution")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Duration vs Mean Probability scatter
    if segment_mean_probs:
        scatter = axes[1].scatter(
            segment_durations,
            segment_mean_probs,
            c=np.arange(len(segment_durations)),
            cmap="viridis",
            s=80,
            alpha=0.7,
        )
        axes[1].set_title("Segment Duration vs Mean Probability")
        axes[1].set_xlabel("Duration (s)")
        axes[1].set_ylabel("Mean Probability")
        plt.colorbar(scatter, ax=axes[1], label="Segment Index")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No probability data available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Duration vs Mean Probability")

    plt.tight_layout()
    distribution_path = output_dir / "segment_distribution.png"
    plt.savefig(str(distribution_path), dpi=150, bbox_inches="tight")
    plt.close()
    console.print(
        f"[green]Distribution plots saved to:[/green] {linkify(distribution_path)}"
    )

# Save concatenated audio if requested
if args.save_concatenated and segments_with_audio:
    console.print("[cyan]Generating concatenated audio...[/cyan]")
    concatenated_parts = []
    silence_marker = np.zeros(
        int(args.silence_marker_duration * SAMPLE_RATE), dtype=np.float32
    )

    for idx, (seg_meta, _) in enumerate(segments_with_audio):
        start_sample = int(seg_meta["start_s"] * SAMPLE_RATE)
        end_sample = int(seg_meta["end_s"] * SAMPLE_RATE)
        start_sample = max(0, start_sample)
        end_sample = min(total_audio_samples, end_sample)
        segment_audio = audio_np[start_sample:end_sample]

        concatenated_parts.append(segment_audio)
        if idx < len(segments_with_audio) - 1:  # Don't add silence after last segment
            concatenated_parts.append(silence_marker)

    if concatenated_parts:
        concatenated_audio = np.concatenate(concatenated_parts)
        concat_path = output_dir / "concatenated_segments.wav"
        sf.write(str(concat_path), concatenated_audio, SAMPLE_RATE)
        console.print(
            f"[green]Concatenated audio saved to:[/green] {linkify(concat_path)}"
        )

        # Save concatenation metadata
        concat_meta = {
            "total_segments": len(segments_with_audio),
            "total_duration_s": float(len(concatenated_audio) / SAMPLE_RATE),
            "silence_marker_duration_s": args.silence_marker_duration,
            "segments": [
                {
                    "index": idx,
                    "start_s": seg_meta["start_s"],
                    "end_s": seg_meta["end_s"],
                    "duration_s": seg_meta["duration_s"],
                }
                for idx, (seg_meta, _) in enumerate(segments_with_audio)
            ],
        }
        with open(output_dir / "concatenation_meta.json", "w", encoding="utf-8") as fh:
            json.dump(concat_meta, fh, ensure_ascii=False, indent=2)

# Save comprehensive summary
summary = {
    "total_segments": len(segments_with_audio),
    "parameters": {
        "min_duration_s": min_duration_s,
        "max_duration_s": max_duration_s,
        "context_window_s": context_window_s,
        "sort_by": args.sort_by,
        "frame_shift_ms": FRAME_SHIFT_MS,
        "sample_rate": SAMPLE_RATE,
    },
    "statistics": {
        "total_audio_duration_s": round(total_duration, 3),
        "mean_segment_duration_s": round(np.mean(segment_durations), 3)
        if segment_durations
        else 0,
        "median_segment_duration_s": round(np.median(segment_durations), 3)
        if segment_durations
        else 0,
        "std_segment_duration_s": round(np.std(segment_durations), 3)
        if segment_durations
        else 0,
        "min_segment_duration_s": round(min(segment_durations), 3)
        if segment_durations
        else 0,
        "max_segment_duration_s": round(max(segment_durations), 3)
        if segment_durations
        else 0,
        "mean_probability": round(np.mean(segment_mean_probs), 4)
        if segment_mean_probs
        else 0,
        "boundary_segments": sum(
            1
            for seg_meta, _ in segments_with_audio
            if seg_meta.get("trough_start") is None
            or seg_meta.get("trough_end") is None
        ),
        "total_troughs_used": len(
            set(
                seg_meta["start_frame"]
                for seg_meta, _ in segments_with_audio
                if seg_meta.get("trough_start")
            )
            | set(
                seg_meta["end_frame"]
                for seg_meta, _ in segments_with_audio
                if seg_meta.get("trough_end")
            )
        ),
    },
    "segments": all_segments_meta,
}

summary_path = output_dir / "trough_to_trough.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)

segs_summary_path = segs_out_dir / "trough_to_trough.json"
with open(segs_summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)

# Print summary table
table = Table(title="Trough-to-Trough Segmentation Summary")
table.add_column("Metric", style="cyan", no_wrap=True)
table.add_column("Value", style="magenta")

table.add_row("Total Segments", str(len(segments_with_audio)))
table.add_row("Total Duration", f"{total_duration:.3f}s")
table.add_row(
    "Mean Duration",
    f"{np.mean(segment_durations):.3f}s" if segment_durations else "N/A",
)
table.add_row(
    "Median Duration",
    f"{np.median(segment_durations):.3f}s" if segment_durations else "N/A",
)
table.add_row(
    "Min Duration", f"{min(segment_durations):.3f}s" if segment_durations else "N/A"
)
table.add_row(
    "Max Duration", f"{max(segment_durations):.3f}s" if segment_durations else "N/A"
)
table.add_row(
    "Boundary Segments",
    str(
        sum(
            1
            for seg_meta, _ in segments_with_audio
            if seg_meta.get("trough_start") is None
            or seg_meta.get("trough_end") is None
        )
    ),
)
table.add_row(
    "Mean Probability",
    f"{np.mean(segment_mean_probs):.4f}" if segment_mean_probs else "N/A",
)
table.add_row("Sort Method", args.sort_by)

console.print(table)

# Print top 5 segments if sorted by non-time criteria
if args.sort_by != "time" and len(segments_with_audio) >= 5:
    console.print("\n[bold cyan]═══ Top 5 Segments ═══[/bold cyan]")
    top_table = Table()
    top_table.add_column("Rank", style="bold yellow", width=6)
    top_table.add_column("Duration", width=10)
    top_table.add_column("Start", width=10)
    top_table.add_column("End", width=10)
    top_table.add_column("Mean Prob", width=10)
    top_table.add_column("Max Prob", width=10)
    top_table.add_column("Type", width=12)

    for i in range(min(5, len(segments_with_audio))):
        seg_meta, _ = segments_with_audio[i]
        is_boundary = (
            seg_meta.get("trough_start") is None or seg_meta.get("trough_end") is None
        )
        seg_type = "Boundary" if is_boundary else "Regular"

        top_table.add_row(
            f"#{i + 1}",
            f"{seg_meta['duration_s']:.3f}s",
            f"{seg_meta['start_s']:.3f}s",
            f"{seg_meta['end_s']:.3f}s",
            f"{seg_meta.get('prob_stats', {}).get('mean', 0):.4f}"
            if seg_meta.get("prob_stats")
            else "N/A",
            f"{seg_meta.get('prob_stats', {}).get('max', 0):.4f}"
            if seg_meta.get("prob_stats")
            else "N/A",
            seg_type,
        )

    console.print(top_table)

console.print(
    f"\n[bold green]Segments saved under:[/bold green] {linkify(segs_out_dir)}"
)
console.print(
    f"[bold green]Trough to trough summary saved to:[/bold green] {linkify(summary_path)}"
)
