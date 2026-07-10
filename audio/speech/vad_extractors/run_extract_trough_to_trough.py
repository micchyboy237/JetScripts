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
from jet.audio.audio_waveform.vad.vad_segment_scorer import (
    VAD_SILENCE_THRESHOLD,
    get_score_components,
    score_vad_segments,
)
from jet.audio.helpers.config import FRAME_LENGTH_MS, FRAME_SHIFT_MS, SAMPLE_RATE
from jet.audio.normalization.dtype_conversion import convert_audio_dtype
from jet.audio.normalization.norm_speech_loudness import normalize_audio_for_vad
from jet.audio.normalization.quant import quantize_audio
from jet.audio.speech.vad_extractors import extract_trough_to_trough, load_probs
from jet.audio.speech.vad_extractors_scoring import (
    BOUNDARY_COLORS,
    CONTENT_COLORS,
    DURATION_COLORS,
    FINAL_SCORE_COLORS,
    format_duration_colored,
    format_duration_score_colored,
    format_score_colored,
    get_quality_label,
)
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
    choices=[
        "time",
        "duration",
        "mean_prob",
        "max_prob",
        "min_prob",
        "final_score",
        "speech_presence",
        "high_confidence",
    ],  # Added new sort options
    help="Sort segments by: time (asc), duration (desc), mean_prob (desc), "
    "max_prob (desc), min_prob (desc), final_score (desc), "
    "speech_presence (desc), high_confidence (desc) (default: time)",
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
    "--vad-score",
    action="store_true",
    default=True,
    help="Generate detailed VAD scoring analysis for each segment (saves vad_info.json).",
)
parser.add_argument(
    "--vad-score-method",
    type=str,
    default="balanced",
    choices=["balanced", "width_heavy", "peak_heavy", "simple"],
    help="Scoring method for VAD analysis (default: balanced).",
)
parser.add_argument(
    "--vad-score-threshold",
    type=float,
    default=VAD_SILENCE_THRESHOLD,
    help=f"Silence threshold for VAD scoring edge trimming (default: {VAD_SILENCE_THRESHOLD}).",
)
parser.add_argument(
    "--vad-score-no-trim",
    action="store_true",
    help="Disable silent edge trimming in VAD scoring analysis.",
)
parser.add_argument(
    "--no-quantize",
    "-nq",
    action="store_true",
    help="Quantize audio to int16 before processing (default: False)",
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
if args.vad_score:
    console.print(
        f"[cyan]VAD scoring enabled: method={args.vad_score_method}, "
        f"threshold={args.vad_score_threshold}, trim={not args.vad_score_no_trim}[/cyan]"
    )

_, audio_np = load_probs(args.audio_path)
display_audio_info(audio_np)
if not args.no_quantize:
    # Always quantize for better segmentation
    # audio_np = convert_audio_dtype(audio_np, "int16")
    audio_np, _ = normalize_audio_for_vad(
        audio_np,
        SAMPLE_RATE,
        # target_rms_db=-20.0,  # Matches V1 default + standard preset
        # max_peak_db=-0.45,  # Exact V1 default peak ceiling
    )
    audio_np, _ = quantize_audio(
        audio_np,
        target_dtype="int16",
        sr=SAMPLE_RATE,
    )

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

# Updated sort options
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
elif args.sort_by == "final_score":
    segments_with_audio.sort(key=lambda x: x[0].get("final_score", 0), reverse=True)
elif args.sort_by == "speech_presence":
    segments_with_audio.sort(
        key=lambda x: x[0].get("scores", {}).get("speech_presence", 0), reverse=True
    )
elif args.sort_by == "high_confidence":
    segments_with_audio.sort(
        key=lambda x: x[0].get("scores", {}).get("high_confidence_ratio", 0),
        reverse=True,
    )

console.print(
    f"[green]Sorted {len(segments_with_audio)} segment(s) by {args.sort_by}[/green]"
)

segs_out_dir = output_dir / "trough_to_trough"
segs_out_dir.mkdir(parents=True, exist_ok=True)

all_segments_meta = []
probs_array = np.array(probs, dtype=float) if probs else np.array([])
n_frames = len(probs_array)
total_audio_samples = len(audio_np) if audio_np is not None else 0
context_samples = int(context_window_s * SAMPLE_RATE)
total_duration = 0.0
segment_durations = []
segment_mean_probs = []


def interpret_vad_score(components: dict) -> dict:
    """
    Provide human-readable interpretation of VAD scoring components.
    """
    if not components:
        return {
            "quality": "unknown",
            "confidence_level": "none",
            "best_use_case": "unknown",
            "recommendations": ["No scoring data available"],
        }
    balanced_score = components.get("balanced_score", 0.0)
    sustained_score = components.get("sustained_score", 0.0)
    peak_score = components.get("peak_score", 0.0)
    if balanced_score >= 0.85:
        quality = "excellent"
        confidence = "high"
    elif balanced_score >= 0.70:
        quality = "good"
        confidence = "moderate"
    elif balanced_score >= 0.55:
        quality = "marginal"
        confidence = "low"
    elif balanced_score >= 0.40:
        quality = "poor"
        confidence = "very_low"
    else:
        quality = "invalid"
        confidence = "none"
    scores = {
        "sustained_speech": sustained_score,
        "keyword_spotting": peak_score,
        "general_vad": balanced_score,
    }
    best_use_case = max(scores, key=scores.get)
    is_sustained = sustained_score > balanced_score
    has_peaks = peak_score > balanced_score
    std_dev = components.get("std_deviation", 1.0)
    recommendations = []
    if quality == "excellent":
        recommendations.append("Segment contains clear, sustained speech")
        recommendations.append("Suitable for transcription or speaker recognition")
    elif quality == "good":
        recommendations.append("Speech is clearly present but may have brief gaps")
        recommendations.append("Review segment boundaries for potential trimming")
    elif quality == "marginal":
        recommendations.append("Speech presence is uncertain - may contain noise")
        recommendations.append("Consider adjusting VAD thresholds")
        recommendations.append("Manual review recommended for critical applications")
    elif quality == "poor":
        recommendations.append("Limited voice activity detected")
        recommendations.append("May be noise, music, or very quiet speech")
        recommendations.append("Consider discarding or lowering confidence thresholds")
    else:
        recommendations.append("No significant voice activity detected")
        recommendations.append("Segment likely contains silence or non-speech audio")
    mean_prob = components.get("mean_probability", 0.0)
    trimmed = components.get("trimmed_segments", 0)
    if trimmed > 0:
        recommendations.append(f"Trimmed {trimmed} silent edge frames")
    if mean_prob < 0.3:
        recommendations.append(
            "Very low mean probability suggests minimal speech content"
        )
    if std_dev > 0.3:
        recommendations.append(
            "High variability indicates inconsistent speech presence"
        )
    return {
        "quality": quality,
        "confidence_level": confidence,
        "best_use_case": best_use_case.replace("_", " "),
        "speech_characteristics": {
            "is_sustained_speech": is_sustained,
            "has_clear_peaks": has_peaks,
            "consistency": "high" if std_dev < 0.3 else "variable",
        },
        "recommendations": recommendations,
    }


for idx, (seg_meta, audio_slice) in enumerate(segments_with_audio):
    seg_dir = segs_out_dir / f"segment_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    enhanced_meta = {
        **seg_meta,
        "rank": idx + 1,
        "sort_by": args.sort_by,
        "original_index": idx,
    }
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(enhanced_meta, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(enhanced_meta)
    total_duration += seg_meta["duration_s"]
    segment_durations.append(seg_meta["duration_s"])
    if seg_meta.get("prob_stats"):
        segment_mean_probs.append(seg_meta["prob_stats"]["mean"])
    if total_audio_samples > 0:
        start_sample = int(seg_meta["start_s"] * SAMPLE_RATE)
        end_sample = int(seg_meta["end_s"] * SAMPLE_RATE)
        start_sample = max(0, start_sample)
        end_sample = min(total_audio_samples, end_sample)
        core_audio = audio_np[start_sample:end_sample]
        if len(core_audio) > 0:
            sf.write(str(seg_dir / "sound.wav"), core_audio, SAMPLE_RATE)
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
        if args.vad_score:
            console.print(f"  [dim]Scoring segment {idx:03d} with VAD scorer...[/dim]")
            score_kwargs = {
                "audio_samples": audio_slice if len(audio_slice) > 0 else None,
                "sample_rate": SAMPLE_RATE,
                "frame_length_ms": FRAME_LENGTH_MS,
                "hop_length_ms": FRAME_SHIFT_MS,
            }
            score_components = get_score_components(
                seg_meta["segment_probs"],
                trim_edges=not args.vad_score_no_trim,
                silence_threshold=args.vad_score_threshold,
                **score_kwargs,
            )
            all_scores = {
                "balanced": score_vad_segments(
                    seg_meta["segment_probs"],
                    method="balanced",
                    trim_edges=not args.vad_score_no_trim,
                    silence_threshold=args.vad_score_threshold,
                    **score_kwargs,
                ),
                "width_heavy": score_vad_segments(
                    seg_meta["segment_probs"],
                    method="width_heavy",
                    trim_edges=not args.vad_score_no_trim,
                    silence_threshold=args.vad_score_threshold,
                    **score_kwargs,
                ),
                "peak_heavy": score_vad_segments(
                    seg_meta["segment_probs"],
                    method="peak_heavy",
                    trim_edges=not args.vad_score_no_trim,
                    silence_threshold=args.vad_score_threshold,
                    **score_kwargs,
                ),
                "simple": score_vad_segments(
                    seg_meta["segment_probs"],
                    method="simple",
                    trim_edges=not args.vad_score_no_trim,
                    silence_threshold=args.vad_score_threshold,
                    **score_kwargs,
                ),
            }
            best_method = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_method]
            vad_info = {
                "segment_index": idx,
                "timestamp": {
                    "start_s": seg_meta["start_s"],
                    "end_s": seg_meta["end_s"],
                    "duration_s": seg_meta["duration_s"],
                },
                "frame_info": {
                    "start_frame": seg_meta["start_frame"],
                    "end_frame": seg_meta["end_frame"],
                    "num_frames": len(seg_meta["segment_probs"]),
                    "frame_shift_ms": FRAME_SHIFT_MS,
                },
                "scoring": {
                    "method_used": args.vad_score_method,
                    "all_scores": all_scores,
                    "best_method": best_method,
                    "best_score": best_score,
                },
                "components": score_components,
                "trim_settings": {
                    "enabled": not args.vad_score_no_trim,
                    "threshold": args.vad_score_threshold,
                    "using_audio_trim": len(audio_slice) > 0,
                },
                "interpretation": interpret_vad_score(score_components)
                if score_components
                else {},
            }
            vad_info_path = seg_dir / "vad_info.json"
            with open(vad_info_path, "w", encoding="utf-8") as fh:
                json.dump(vad_info, fh, ensure_ascii=False, indent=2, default=float)
            quality_color = {
                "excellent": "green",
                "good": "blue",
                "marginal": "yellow",
                "poor": "orange1",
                "invalid": "red",
            }.get(vad_info["interpretation"].get("quality", "unknown"), "white")
            console.print(
                f"    [{quality_color}]VAD Score: {best_score:.3f} ({best_method}) - "
                f"Quality: {vad_info['interpretation'].get('quality', 'N/A')}[/{quality_color}]"
            )

    # Plot generation

    if n_frames > 0:
        f_start = max(0, seg_meta["start_frame"] - 20)
        f_end = min(n_frames, seg_meta["end_frame"] + 21)
        frames = np.arange(f_start, f_end)
        zoomed = probs_array[f_start:f_end]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, zoomed, "b-", linewidth=2, label="VAD Probability", alpha=0.8)
        ax.axvspan(
            seg_meta["start_frame"],
            seg_meta["end_frame"],
            alpha=0.12,
            color="purple",
            label="trough span",
        )

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

# [Previous plot generation code would continue here...]

# Updated score extraction with new field names
if segments_with_audio:
    final_scores = [
        s[0].get("final_score", 0.0)
        for s in segments_with_audio
        if s[0].get("final_score") is not None
    ]
    speech_presence_scores = [
        s[0].get("scores", {}).get("speech_presence", 0.0) for s in segments_with_audio
    ]
    core_density_scores = [
        s[0].get("scores", {}).get("core_speech_density", 0.0)
        for s in segments_with_audio
    ]
    high_confidence_scores = [
        s[0].get("scores", {}).get("high_confidence_ratio", 0.0)
        for s in segments_with_audio
    ]
    continuity_scores = [
        s[0].get("scores", {}).get("speech_continuity", 0.0)
        for s in segments_with_audio
    ]
    boundary_scores = [
        s[0].get("scores", {}).get("boundary_quality_score", 0.0)
        for s in segments_with_audio
    ]
    duration_scores_list = [
        s[0].get("scores", {}).get("duration_score", 0.0) for s in segments_with_audio
    ]

    if final_scores:
        quality_bins = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "bad": 0}
        for fs in final_scores:
            if fs >= 0.80:
                quality_bins["excellent"] += 1
            elif fs >= 0.60:
                quality_bins["good"] += 1
            elif fs >= 0.40:
                quality_bins["fair"] += 1
            elif fs >= 0.20:
                quality_bins["poor"] += 1
            else:
                quality_bins["bad"] += 1

        score_summary_table = Table(
            title="Score Distribution Summary",
            header_style="bold white",
            border_style="bright_black",
        )
        score_summary_table.add_column(
            "Metric", style="cyan", width=20
        )  # Wider for new names
        score_summary_table.add_column("Mean", width=8, justify="right")
        score_summary_table.add_column("Median", width=8, justify="right")
        score_summary_table.add_column("Min", width=8, justify="right")
        score_summary_table.add_column("Max", width=8, justify="right")

        score_summary_table.add_row(
            "Final Score",
            f"{np.mean(final_scores):.3f}",
            f"{np.median(final_scores):.3f}",
            f"{np.min(final_scores):.3f}",
            f"{np.max(final_scores):.3f}",
        )
        score_summary_table.add_row(
            "Speech Presence",  # NEW: replaces median_prob_score
            f"{np.mean(speech_presence_scores):.3f}",
            f"{np.median(speech_presence_scores):.3f}",
            f"{np.min(speech_presence_scores):.3f}",
            f"{np.max(speech_presence_scores):.3f}",
        )
        score_summary_table.add_row(
            "Core Density",  # Updated name
            f"{np.mean(core_density_scores):.3f}",
            f"{np.median(core_density_scores):.3f}",
            f"{np.min(core_density_scores):.3f}",
            f"{np.max(core_density_scores):.3f}",
        )
        score_summary_table.add_row(
            "High Confidence %",  # NEW: replaces peak_strength
            f"{np.mean(high_confidence_scores):.3f}",
            f"{np.median(high_confidence_scores):.3f}",
            f"{np.min(high_confidence_scores):.3f}",
            f"{np.max(high_confidence_scores):.3f}",
        )
        score_summary_table.add_row(
            "Continuity",  # Updated name
            f"{np.mean(continuity_scores):.3f}",
            f"{np.median(continuity_scores):.3f}",
            f"{np.min(continuity_scores):.3f}",
            f"{np.max(continuity_scores):.3f}",
        )
        score_summary_table.add_row(
            "Boundary",
            f"{np.mean(boundary_scores):.3f}",
            f"{np.median(boundary_scores):.3f}",
            f"{np.min(boundary_scores):.3f}",
            f"{np.max(boundary_scores):.3f}",
        )
        score_summary_table.add_row(
            "Duration",
            f"{np.mean(duration_scores_list):.3f}",
            f"{np.median(duration_scores_list):.3f}",
            f"{np.min(duration_scores_list):.3f}",
            f"{np.max(duration_scores_list):.3f}",
        )
        console.print(score_summary_table)

        quality_dist_table = Table(
            title="Quality Distribution",
            header_style="bold white",
            border_style="bright_black",
        )
        quality_dist_table.add_column("Quality", width=12)
        quality_dist_table.add_column("Count", width=8, justify="right")
        quality_dist_table.add_column("Percent", width=10, justify="right")
        quality_dist_table.add_column("Bar", width=30)
        colors = {
            "excellent": "green",
            "good": "bright_green",
            "fair": "yellow",
            "poor": "orange1",
            "bad": "red",
        }
        total_segs = len(final_scores)
        for quality, count in quality_bins.items():
            pct = (count / total_segs) * 100 if total_segs > 0 else 0
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            quality_dist_table.add_row(
                f"[{colors[quality]}]{quality.capitalize()}[/{colors[quality]}]",
                str(count),
                f"{pct:.1f}%",
                f"[{colors[quality]}]{bar}[/{colors[quality]}]",
            )
        console.print(quality_dist_table)

    # Updated summary table
    summary_table = Table(title="Trough-to-Trough Segmentation Summary")
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="magenta")
    summary_table.add_row("Total Segments", str(len(segments_with_audio)))
    summary_table.add_row("Total Duration", f"{total_duration:.3f}s")
    summary_table.add_row(
        "Mean Duration",
        f"{np.mean(segment_durations):.3f}s" if segment_durations else "N/A",
    )
    summary_table.add_row(
        "Median Duration",
        f"{np.median(segment_durations):.3f}s" if segment_durations else "N/A",
    )
    summary_table.add_row(
        "Min Duration", f"{min(segment_durations):.3f}s" if segment_durations else "N/A"
    )
    summary_table.add_row(
        "Max Duration", f"{max(segment_durations):.3f}s" if segment_durations else "N/A"
    )
    summary_table.add_row(
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
    summary_table.add_row(
        "Mean Probability",
        f"{np.mean(segment_mean_probs):.4f}" if segment_mean_probs else "N/A",
    )
    summary_table.add_row("Sort Method", args.sort_by)
    if final_scores:
        summary_table.add_row(
            "Mean Final Score",
            f"{np.mean(final_scores):.3f}",
        )
        summary_table.add_row(
            "Mean Speech Presence",  # NEW
            f"{np.mean(speech_presence_scores):.3f}",
        )
        summary_table.add_row(
            "Mean Core Density",  # Updated
            f"{np.mean(core_density_scores):.3f}",
        )
        summary_table.add_row(
            "Mean High Confidence %",  # NEW
            f"{np.mean(high_confidence_scores):.3f}",
        )
        summary_table.add_row(
            "Mean Continuity",  # Updated
            f"{np.mean(continuity_scores):.3f}",
        )
        summary_table.add_row(
            "Mean Boundary Score",
            f"{np.mean(boundary_scores):.3f}",
        )
    console.print(summary_table)

    console.print(
        "\n[bold cyan]═══ Trough-to-Trough Segmentation Results ═══[/bold cyan]"
    )

    # Updated results table with new column names and order
    results_table = Table(
        title=f"Segment Overview · {len(segments_with_audio)} segments · sorted by {args.sort_by}",
        show_lines=False,
        header_style="bold white",
        border_style="bright_black",
    )
    results_table.add_column("#", style="dim", width=4, justify="right")
    results_table.add_column("Start", width=9, justify="right")
    results_table.add_column("End", width=9, justify="right")
    results_table.add_column("Dur (s)", width=8, justify="right")
    results_table.add_column("Dur Score", width=9, justify="right")
    results_table.add_column(
        "Presence", width=9, justify="right"
    )  # NEW: Speech Presence
    results_table.add_column("Core", width=8, justify="right")  # Core Density
    results_table.add_column("HC%", width=8, justify="right")  # High Confidence %
    results_table.add_column("Cont", width=8, justify="right")  # Continuity
    results_table.add_column("Boundary", width=9, justify="right")
    results_table.add_column("Final", width=7, justify="right", style="bold yellow")
    results_table.add_column("Quality", width=10)
    results_table.add_column("Files", width=10, justify="center")

    for idx, (seg_meta, _) in enumerate(segments_with_audio):
        scores = seg_meta.get("scores", {})
        final_score = seg_meta.get("final_score", 0.0)
        prob_stats = seg_meta.get("prob_stats", {})
        quality_label, quality_color = get_quality_label(final_score)
        seg_dir = segs_out_dir / f"segment_{idx:03d}"
        wav_path = seg_dir / "sound.wav"
        folder_link = f"[link=file://{seg_dir.resolve()}][bold bright_blue]📁[/bold bright_blue][/link]"
        play_link = f"[link=file://{wav_path.resolve()}][bold bright_green]▶[/bold bright_green][/link]"

        results_table.add_row(
            str(idx),
            f"{seg_meta['start_s']:.2f}",
            f"{seg_meta['end_s']:.2f}",
            format_duration_colored(
                seg_meta["duration_s"],
                DURATION_COLORS["very_short_max"],
                DURATION_COLORS["short_max"],
                DURATION_COLORS["optimal_max"],
                DURATION_COLORS["long_max"],
            ),
            format_duration_score_colored(
                scores.get("duration_score", 0) if scores else 0,
                seg_meta["duration_s"],
            ),
            format_score_colored(
                scores.get("speech_presence", 0) if scores else 0,  # NEW
                CONTENT_COLORS["red_max"],
                CONTENT_COLORS["yellow_max"],
            ),
            format_score_colored(
                scores.get("core_speech_density", 0) if scores else 0,
                CONTENT_COLORS["red_max"],
                CONTENT_COLORS["yellow_max"],
            ),
            format_score_colored(
                scores.get("high_confidence_ratio", 0) if scores else 0,  # NEW
                CONTENT_COLORS["red_max"],
                CONTENT_COLORS["yellow_max"],
            ),
            format_score_colored(
                scores.get("speech_continuity", 0) if scores else 0,
                CONTENT_COLORS["red_max"],
                CONTENT_COLORS["yellow_max"],
            ),
            format_score_colored(
                scores.get("boundary_quality_score", 0) if scores else 0,
                BOUNDARY_COLORS["red_max"],
                BOUNDARY_COLORS["yellow_max"],
            ),
            format_score_colored(
                final_score if scores else 0,
                FINAL_SCORE_COLORS["red_max"],
                FINAL_SCORE_COLORS["yellow_max"],
            ),
            f"[{quality_color}]{quality_label}[/{quality_color}]",
            f"{folder_link}  {play_link}",
        )

    console.print(results_table)
    console.print(
        f"\n[bold green]Segments saved under:[/bold green] {linkify(segs_out_dir)}"
    )
    console.print(
        f"[bold green]Trough to trough summary saved to:[/bold green] {linkify(output_dir / 'trough_to_trough.json')}"
    )
