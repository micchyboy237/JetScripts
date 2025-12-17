# JetScripts/audio/run_record_mic.py
import os
from datetime import datetime
import shutil
from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.record_mic_speech_detection import record_from_mic
from jet.file.utils import save_file
import copy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    duration_seconds = 60
    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Record with trim_silent=True → returns trimmed np.ndarray directly
    data_stream = record_from_mic(duration_seconds, trim_silent=True, output_dir=OUTPUT_DIR)
    segments = []
    for seg_meta, seg_audio_np, full_audio_np in data_stream:
        meta = copy.deepcopy(seg_meta)
        # Replace these keys with their value in seconds (3 decimals)
        for key in [
            "original_start_sample",
            "original_end_sample",
            "saved_start_sample",
            "saved_end_sample"
        ]:
            if key in meta:
                meta[key] = round(meta[key] / SAMPLE_RATE, 3)
        segments.append(meta)
        save_file(segments, f"{OUTPUT_DIR}/all_segments.json", verbose=False)
