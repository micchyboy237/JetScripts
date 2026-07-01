import argparse
import json
import shutil
from pathlib import Path

from jet.audio.audio_waveform.vad.vad_logging import linkify
from jet.audio.helpers.config import SAMPLE_RATE
from jet.audio.speech_handlers.vad_firered_splitter import split_segment_with_vad
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
    default=1.0,
    help="Minimum segment duration in seconds (default: 1.0)",
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

sub_segments = split_segment_with_vad(
    audio_np,
)

sub_segments_path = output_dir / "sub_segments.json"
with open(sub_segments_path, "w", encoding="utf-8") as fh:
    json.dump(sub_segments, fh, ensure_ascii=False, indent=2)
console.print(
    f"[bold green]Sub-segments saved under:[/bold green] {linkify(sub_segments_path)}"
)
