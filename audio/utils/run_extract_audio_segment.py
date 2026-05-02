import argparse
import shutil
from pathlib import Path

import soundfile as sf
from jet.audio.helpers.energy_base import get_audio_duration
from jet.audio.norm.norm_speech_loudness import normalize_speech_loudness
from jet.audio.utils import extract_audio_segment
from jet.audio.utils.loader import load_audio

DEFAULT_INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

parser = argparse.ArgumentParser(description="Extract an audio segment from a file.")
parser.add_argument(
    "input_audio",
    nargs="?",
    default=DEFAULT_INPUT_AUDIO,
    help=f"Path to input audio file (default: {DEFAULT_INPUT_AUDIO})",
)
parser.add_argument(
    "-s",
    "--start",
    type=float,
    default=0.0,
    help="Start time in seconds (default: 0.0)",
)
parser.add_argument(
    "-e",
    "--end",
    type=float,
    default=None,
    help="End time in seconds (default: None = end of audio)",
)
args = parser.parse_args()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_AUDIO = args.input_audio

target_sr = 16000
start = args.start
end = args.end

if end is None:
    audio_np, _ = load_audio(INPUT_AUDIO, target_sr)
    end = get_audio_duration(audio_np, target_sr)

# Extract from raw input audio
segment, sr = extract_audio_segment(INPUT_AUDIO, start=start, end=end)
output_path = OUTPUT_DIR / "extracted_audio.wav"
sf.write(output_path, segment, sr)

# Extract from normalized audio
norm_segment = normalize_speech_loudness(segment, sr)
norm_output_path = OUTPUT_DIR / "extracted_audio_norm.wav"
sf.write(norm_output_path, norm_segment, sr)

print("Extracted from raw input. Saved at:")
print(output_path)
print("Extracted from normalized input. Saved at:")
print(norm_output_path)
