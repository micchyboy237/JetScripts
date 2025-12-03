import json
from jet.audio.speech.silero.speech_utils import extract_speech_timestamps
from jet.audio.utils import resolve_audio_paths
from jet.logger import logger
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    files = resolve_audio_paths(audio_file)

    for num, f in enumerate(files, start=1):
        sub_dir = os.path.basename(f)
        speech_timestamps = extract_speech_timestamps(f)
        logger.info(f"\nFile {num}: {sub_dir}")
        logger.success(json.dumps(speech_timestamps, indent=2))
        save_file(speech_timestamps, f"{OUTPUT_DIR}/{sub_dir}/speech_timestamps.json", verbose=False)