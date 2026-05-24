import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf
from jet.audio.audio_waveform.vad.vad_logging import linkify
from jet.audio.helpers.loudness import normalize_loudness
from jet.audio.normalization.norm_speech_loudness import (
    normalize_audio_for_vad,
    normalize_speech_loudness,
)
from jet.audio.normalization.norm_speech_loudness_firered import (
    normalize_speech_loudness as normalize_speech_loudness_firered,
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
loudness_norm = normalize_loudness(audio_np, sr)
louder_audio = loudness_norm.normalized_data
loudness_norm_stats = loudness_norm.get_stats()
norm_audio_vad, norm_audio_vad_stats = normalize_audio_for_vad(audio_np, sr)
norm_audio_sil = normalize_speech_loudness(audio_np, sr)
norm_audio_fir = normalize_speech_loudness_firered(audio_np, sr)

# Save normalized audio files
output_path_orig = OUTPUT_DIR / "orig_sound.wav"
sf.write(output_path_orig, audio_np, sr)

output_path_louder = OUTPUT_DIR / "louder_audio.wav"
sf.write(output_path_louder, louder_audio, sr)
output_path_louder_info = OUTPUT_DIR / "louder_info.json"
with open(output_path_louder_info, "w", encoding="utf-8") as f:
    json.dump(loudness_norm_stats, f, indent=2, ensure_ascii=False)

output_path_vad = OUTPUT_DIR / "normalized_audio_vad.wav"
sf.write(output_path_vad, norm_audio_vad, sr)

output_path_vad_info = OUTPUT_DIR / "norm_vad_info.json"
with open(output_path_vad_info, "w", encoding="utf-8") as f:
    json.dump(norm_audio_vad_stats, f, indent=2, ensure_ascii=False)

output_path_sil = OUTPUT_DIR / "normalized_audio_sil.wav"
sf.write(output_path_sil, norm_audio_sil, sr)

output_path_fir = OUTPUT_DIR / "normalized_audio_fir.wav"
sf.write(output_path_fir, norm_audio_fir, sr)


# Print summary using console.print
console.print("Normalized audio saved at:")
console.print(f"[dim green] Orig → {linkify(output_path_orig)}[/dim green]")
console.print(f"[dim green] Louder → {linkify(output_path_louder)}[/dim green]")
console.print(
    f"[dim green] Louder Info → {linkify(output_path_louder_info)}[/dim green]"
)
console.print(f"[dim green] VAD → {linkify(output_path_vad)}[/dim green]")
console.print(f"[dim green] VAD Info → {linkify(output_path_vad_info)}[/dim green]")
console.print(f"[dim green] SIL → {linkify(output_path_sil)}[/dim green]")
console.print(f"[dim green] FIR → {linkify(output_path_fir)}[/dim green]")
