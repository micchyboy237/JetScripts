import os
from datetime import datetime
from jet.audio.helpers.silence import trim_silent_chunks
from jet.audio.record_mic import record_from_mic, save_wav_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
OUTPUT_SUFFIX = datetime.now().strftime('%Y%m%d_%H%M%S')

if __name__ == "__main__":
    duration_seconds = 20  # Change to how long you want to record
    data = record_from_mic(duration_seconds)

    output_file = f"{OUTPUT_DIR}/recording_{OUTPUT_SUFFIX}.wav"
    save_wav_file(output_file, data)

    output_file = f"{OUTPUT_DIR}/recording_trimmed_{OUTPUT_SUFFIX}.wav"
    save_wav_file(output_file, trim_silent_chunks(data))
