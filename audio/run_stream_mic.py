import os
from datetime import datetime
import shutil
import numpy as np
from typing import List
from pathlib import Path
from jet.audio.record_mic import save_wav_file, SAMPLE_RATE
from jet.audio.stream_mic import stream_non_silent_audio
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def save_chunk(chunk: np.ndarray, chunk_index: int, timestamp: str) -> str:
    """Save an individual audio chunk to a WAV file."""
    chunk_filename = f"{OUTPUT_DIR}/stream_chunk_{timestamp}_{chunk_index:04d}.wav"
    save_wav_file(chunk_filename, chunk)
    chunk_duration = len(chunk) / SAMPLE_RATE
    logger.debug(
        f"Saved chunk {chunk_index} to {chunk_filename}, size: {len(chunk)} samples, duration: {chunk_duration:.2f}s")
    return chunk_filename


def main():
    """
    Stream non-silent audio from microphone and save each chunk to individual WAV files.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chunk_index = 0
    saved_files: List[str] = []
    total_samples = 0
    min_chunk_duration = 1.0
    overlap_duration = 0.2

    for chunk in stream_non_silent_audio(
        silence_threshold=None,
        chunk_duration=0.5,
        silence_duration=2.0,
        min_chunk_duration=min_chunk_duration,
        overlap_duration=overlap_duration
    ):
        chunk_duration = len(chunk) / SAMPLE_RATE
        expected_min_duration = min_chunk_duration + overlap_duration
        if chunk_duration < expected_min_duration:
            logger.warning(
                f"Chunk {chunk_index} duration {chunk_duration:.2f}s is less than minimum {expected_min_duration:.2f}s")
        if chunk.shape[0] % int(SAMPLE_RATE * 0.5) != 0 and chunk.shape[0] < int(SAMPLE_RATE * expected_min_duration):
            logger.warning(
                f"Chunk {chunk_index} has non-standard size: {chunk.shape[0]} samples")
        chunk_filename = save_chunk(chunk, chunk_index, timestamp)
        saved_files.append(chunk_filename)
        total_samples += chunk.shape[0]
        print(f"Saved chunk {chunk_index} to {chunk_filename}, samples: {chunk.shape[0]}, duration: {chunk_duration:.2f}s, "
              f"overlap: {overlap_duration:.2f}s")
        chunk_index += 1

    if not saved_files:
        print("No non-silent audio chunks captured.")
        return

    total_duration = total_samples / SAMPLE_RATE
    print(f"Streamed audio saved to {len(saved_files)} chunk files in {OUTPUT_DIR}, "
          f"total duration: {total_duration:.2f}s (including overlaps)")


if __name__ == "__main__":
    main()
