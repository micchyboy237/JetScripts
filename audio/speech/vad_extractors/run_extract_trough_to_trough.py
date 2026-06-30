import argparse
import json
import shutil
from pathlib import Path

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
args = parser.parse_args()
min_valley_duration = 0.25
frame_offset = 0
smoothing_window = 0
output_dir = Path(args.output_dir)
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)
audio_np, _ = load_audio(args.audio_path, sr=SAMPLE_RATE, mono=True)


# Updated call: pass audio directly, function handles everything internally
segments_with_audio, _ = extract_trough_to_trough(
    probs_or_audio=audio_np,
    frame_shift_ms=FRAME_SHIFT_MS,
    sample_rate=SAMPLE_RATE,
    with_audio=True,
    with_scores=True,  # Enable scores to get per-segment probs
)

segs_out_dir = output_dir / "trough_to_trough"
segs_out_dir.mkdir(parents=True, exist_ok=True)
all_segments_meta = []
for idx, (seg_meta, audio_slice) in enumerate(segments_with_audio):
    seg_dir = segs_out_dir / f"segment_{idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(seg_meta, fh, ensure_ascii=False, indent=2)
    all_segments_meta.append(seg_meta)
    if len(audio_slice) > 0:
        sf.write(str(seg_dir / "sound.wav"), audio_slice, SAMPLE_RATE)
    # Save per-segment probability scores as plain array
    if seg_meta.get("segment_probs"):
        with open(seg_dir / "probs.json", "w", encoding="utf-8") as fh:
            json.dump(seg_meta["segment_probs"], fh, ensure_ascii=False, indent=2)
        # Save probability info (frame_shift, frame_start, frame_end, stats)
        if seg_meta.get("prob_stats"):
            probs_info = {
                "frame_shift_sec": FRAME_SHIFT_MS / 1000.0,
                "frame_start": seg_meta["start_frame"],
                "frame_end": seg_meta["end_frame"],
                "stats": seg_meta["prob_stats"],
            }
            with open(seg_dir / "probs_info.json", "w", encoding="utf-8") as fh:
                json.dump(probs_info, fh, ensure_ascii=False, indent=2)
summary_path = output_dir / "trough_to_trough.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(all_segments_meta, fh, ensure_ascii=False, indent=2)
console.print(f"[bold green]Segments save under:[/bold green] {linkify(segs_out_dir)}")
console.print(
    f"[bold green]Trough to trough summary saved to:[/bold green] {linkify(summary_path)}"
)
