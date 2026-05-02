import argparse
import shutil
from pathlib import Path

from jet.audio.speech.firered.speech_waves import (
    check_speech_waves,
)
from jet.file.utils import load_file, save_file

DEFAULT_SEGMENT_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/audio_waveform/generated/speech_tracking/saved_speech_segments/segment_20260502_221930_278"

parser = argparse.ArgumentParser(
    description="Check speech waves from segment dir and settings."
)
parser.add_argument(
    "segment_dir",
    nargs="?",
    default=DEFAULT_SEGMENT_DIR,
    help="Directory containing the speech segment probs files",
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

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

segment_dir = args.segment_dir
speech_probs_file = Path(segment_dir) / "probs.json"
hybrid_speech_probs_file = Path(segment_dir) / "hybrid_probs.json"

speech_probs = load_file(str(speech_probs_file))
hybrid_speech_probs = load_file(str(hybrid_speech_probs_file))
sample_rate = args.sample_rate
threshold = args.threshold

speech_waves = check_speech_waves(
    speech_probs=speech_probs,
    threshold=threshold,
    sampling_rate=sample_rate,
)
hybrid_speech_waves = check_speech_waves(
    speech_probs=hybrid_speech_probs,
    threshold=threshold,
    sampling_rate=sample_rate,
)

save_file(speech_waves, OUTPUT_DIR / "speech_waves.json")
save_file(hybrid_speech_waves, OUTPUT_DIR / "hybrid_speech_waves.json")
