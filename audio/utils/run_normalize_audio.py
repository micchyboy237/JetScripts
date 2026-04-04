import argparse
import shutil
from pathlib import Path

import soundfile as sf
from jet.audio.norm.norm_speech_loudness import normalize_speech_loudness
from jet.audio.utils.loader import load_audio

DEFAULT_INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

parser = argparse.ArgumentParser(description="Normalize audio loudness.")
parser.add_argument(
    "input_audio",
    nargs="?",
    default=DEFAULT_INPUT_AUDIO,
    help=f"Path to input audio file (default: {DEFAULT_INPUT_AUDIO})",
)
args = parser.parse_args()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_AUDIO = args.input_audio
target_sr = 16000

# Load full audio
audio_np, sr = load_audio(INPUT_AUDIO, target_sr)

# Normalize
norm_audio = normalize_speech_loudness(audio_np, sr)

# Save
output_path = OUTPUT_DIR / "normalized_audio.wav"
sf.write(output_path, norm_audio, sr)

print("Normalized audio saved at:")
print(output_path)
