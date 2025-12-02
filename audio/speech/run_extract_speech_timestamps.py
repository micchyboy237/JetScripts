import json
import os
import shutil

from jet.audio.speech.silero.speech_utils import extract_speech_timestamps
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    speech_timestamps = extract_speech_timestamps(audio_file)
    logger.success(json.dumps(speech_timestamps, indent=2))  # [{'start': 0.0, 'end': 2.5}, {'start': 5.0, 'end': 7.8}]
    save_file(speech_timestamps, f"{OUTPUT_DIR}/speech_timestamps.json")