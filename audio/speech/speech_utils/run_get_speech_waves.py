import shutil
from pathlib import Path

# from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps
# from jet.audio.speech.silero.speech_utils import get_speech_waves
from jet.audio.speech.firered.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.firered.speech_waves import get_speech_waves
from jet.file.utils import load_file, save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/audio_waveform/generated/speech_tracking/saved_speech_segments/segment_20260502_221935_279/sound.wav"

speech_probs = load_file(audio_file)
sample_rate = 16000
threshold = 0.3

segments, scores = extract_speech_timestamps(
    audio_file,
    threshold=threshold,
    with_scores=True,
)

speech_waves = get_speech_waves(audio_file, scores, threshold=threshold)

save_file(segments, OUTPUT_DIR / "segments.json")
save_file(speech_waves, OUTPUT_DIR / "speech_waves.json")
