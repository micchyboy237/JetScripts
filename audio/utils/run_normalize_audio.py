import argparse
import shutil
from pathlib import Path

import librosa
import soundfile as sf

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

# Load audio with librosa
audio_np, sr = librosa.load(INPUT_AUDIO, sr=None, mono=False)

# ==================== Normalization using librosa ====================

# Option 1: Peak normalization (most common & simple)
norm_audio = librosa.util.normalize(audio_np, norm=inf=-1.0)   # -1 dBFS peak

# Option 2: RMS normalization to a target level (uncomment if preferred)
# target_rms = 0.2
# norm_audio = audio_np * (target_rms / (librosa.feature.rms(y=audio_np)[0].mean() + 1e-8))

# Option 3: Normalize to a specific max dB (e.g. -3 dBFS)
# max_db = -3.0
# peak = np.max(np.abs(audio_np))
# norm_audio = audio_np * (10 ** (max_db / 20) / (peak + 1e-8))

# ===================================================================

# Save normalized audio
output_path = OUTPUT_DIR / "normalized_audio.wav"
sf.write(output_path, norm_audio.T if audio_np.ndim > 1 else norm_audio, sr)

print("Normalized audio saved at:")
print(output_path)