# JetScripts/audio/run_record_mic.py
import os
from datetime import datetime
from jet.audio.record_mic import record_from_mic
from jet.audio.speech.wav_utils import save_wav_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_SUFFIX = datetime.now().strftime('%Y%m%d_%H%M%S')

if __name__ == "__main__":
    duration_seconds = 25

    # Record with trim_silent=True → returns trimmed np.ndarray directly
    data = record_from_mic(duration_seconds, trim_silent=True)
    if data is None:
        print("No sound detected or recording failed.")
    else:
        output_file = f"{OUTPUT_DIR}/recording_trimmed_{OUTPUT_SUFFIX}.wav"
        save_wav_file(output_file, data)
        print(f"Trimmed recording saved: {output_file}")

    # # Also save untrimmed version for comparison
    # raw_data = record_from_mic(duration_seconds, trim_silent=False)
    # if raw_data is not None:
    #     raw_file = f"{OUTPUT_DIR}/recording_raw_{OUTPUT_SUFFIX}.wav"
    #     save_wav_file(raw_file, raw_data)
    #     print(f"Raw recording saved: {raw_file}")