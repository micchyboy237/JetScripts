import argparse
import shutil
from pathlib import Path

from jet.audio.speech.firered.speech_waves import check_speech_waves
from jet.file.utils import load_file, save_file

DEFAULT_PROBS_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/vad_loaders/generated/run_load_vad_hybrid_probs/segments/segment_002/hybrid_probs.json"

parser = argparse.ArgumentParser(
    description="Check speech waves from a single probs.json file."
)

parser.add_argument(
    "probs_path",
    nargs="?",
    default=DEFAULT_PROBS_PATH,
    help="Path to the probs.json file",
)

parser.add_argument(
    "-sr",
    "--sample_rate",
    type=int,
    default=16000,
    help="Sample rate of the audio (default: 16000)",
)

parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.3,
    help="Speech probability threshold (default: 0.3)",
)

args = parser.parse_args()

# Output directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load probabilities
probs_path = Path(args.probs_path)
speech_probs = load_file(str(probs_path))

sample_rate = args.sample_rate
threshold = args.threshold

# Run check
speech_waves = check_speech_waves(
    speech_probs=speech_probs,
    threshold=threshold,
    sampling_rate=sample_rate,
)

# Save result
output_file = OUTPUT_DIR / "speech_waves.json"
save_file(speech_waves, output_file)

print(f"✅ Done! Speech waves saved to: {output_file}")
print(f"   Found {len(speech_waves)} speech segments")
