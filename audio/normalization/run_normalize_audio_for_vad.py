import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf
from jet.audio.audio_waveform.vad.vad_logging import linkify
from jet.audio.normalization.norm_audio import normalize_audio_for_vad
from jet.audio.normalization.norm_speech_loudness import normalize_speech_loudness
from jet.audio.normalization.norm_speech_loudness_firered import (
    normalize_speech_loudness as normalize_speech_loudness_fr,
)
from jet.audio.utils.loader import load_audio
from rich.console import Console

console = Console()

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
norm_audio_vad, norm_vad_info = normalize_audio_for_vad(audio_np, sr)
norm_audio_sil = normalize_speech_loudness(audio_np, sr)
norm_audio_fir = normalize_speech_loudness_fr(audio_np, sr)

# Save normalized audio files
output_path_vad = OUTPUT_DIR / "normalized_audio_vad.wav"
sf.write(output_path_vad, norm_audio_vad, sr)

output_path_sil = OUTPUT_DIR / "normalized_audio_sil.wav"
sf.write(output_path_sil, norm_audio_sil, sr)

output_path_fir = OUTPUT_DIR / "normalized_audio_fir.wav"
sf.write(output_path_fir, norm_audio_fir, sr)

# Save VAD info as JSON
output_path_vad_info = OUTPUT_DIR / "norm_vad_info.json"
with open(output_path_vad_info, "w", encoding="utf-8") as f:
    json.dump(norm_vad_info, f, indent=2, ensure_ascii=False)

# Print summary using console.print
console.print("Normalized audio saved at:")
console.print(f"[dim green] VAD Info → {linkify(output_path_vad_info)}[/dim green]")
console.print(f"[dim green] VAD → {linkify(output_path_vad)}[/dim green]")
console.print(f"[dim green] SIL → {linkify(output_path_sil)}[/dim green]")
console.print(f"[dim green] FIR → {linkify(output_path_fir)}[/dim green]")
