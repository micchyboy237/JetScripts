import os
from datetime import datetime
from jet.audio.record_mic import record_from_mic, save_wav_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

OUTPUT_FILE = f"{OUTPUT_DIR}/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

if __name__ == "__main__":
    duration_seconds = 100  # Change to how long you want to record
    data = record_from_mic(duration_seconds)
    save_wav_file(OUTPUT_FILE, data)
