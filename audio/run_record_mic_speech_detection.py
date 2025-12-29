# JetScripts/audio/run_record_mic.py
import json
from pathlib import Path
import shutil
import numpy as np

from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.record_mic_speech_detection import record_from_mic
from jet.audio.speech.silero.speech_types import SpeechSegment
from jet.audio.speech.wav_utils import save_wav_file
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_segment_data(speech_seg: SpeechSegment, seg_audio_np: np.ndarray):
    segment_root = Path(OUTPUT_DIR) / "segments"
    segment_root.mkdir(parents=True, exist_ok=True)

    # Find next available segment directory name
    # (segment_001, segment_002, ...)
    existing = sorted(segment_root.glob("segment_*"))
    used_numbers = set()
    for seg in existing:
        try:
            used_numbers.add(int(seg.name.split("_")[1]))
        except Exception:
            continue

    # Pick smallest unused positive integer for segment id/dir
    seg_number = 1
    while seg_number in used_numbers:
        seg_number += 1

    seg_dir = segment_root / f"segment_{seg_number:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    wav_path = seg_dir / "sound.wav"
    metadata_path = seg_dir / "metadata.json"

    seg_sound_file = save_wav_file(wav_path, seg_audio_np)
    metadata_path.write_text(json.dumps(speech_seg, indent=2), encoding="utf-8")

    logger.success(f"Segment {seg_number} data saved to:")
    logger.success(seg_sound_file, bright=True)
    logger.success(metadata_path, bright=True)

if __name__ == "__main__":
    duration_seconds = None
    trim_silent = True
    quit_on_silence = False

    # Record with trim_silent=True → returns trimmed np.ndarray directly
    data_stream = record_from_mic(duration_seconds, trim_silent=trim_silent, quit_on_silence=quit_on_silence)
    segments: list[dict] = []
    pending_segments: dict[int, dict] = {}

    for speech_seg, seg_audio_np, full_audio_np in data_stream:
        # Convert sample-based times to seconds for JSON metadata
        speech_seg_copy = speech_seg.copy()
        for key in [
            "start",
            "end",
        ]:
            if key in speech_seg_copy:
                speech_seg_copy[key] = round(speech_seg_copy[key] / SAMPLE_RATE, 3)

        save_segment_data(speech_seg_copy, seg_audio_np)

        segments.append(speech_seg_copy)
        save_file(segments, OUTPUT_DIR / "all_segments.json", verbose=False)


        output_file = f"{OUTPUT_DIR}/full_recording.wav"
        save_wav_file(output_file, full_audio_np)