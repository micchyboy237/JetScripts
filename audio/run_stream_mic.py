import os
from datetime import datetime
import numpy as np
from typing import List
from pathlib import Path
from jet.audio.record_mic import save_wav_file
from jet.audio.stream_mic import stream_non_silent_audio

from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)


def save_chunk(chunk: np.ndarray, chunk_index: int, timestamp: str) -> str:
    """Save an individual audio chunk to a WAV file."""
    chunk_filename = f"{OUTPUT_DIR}/stream_chunk_{timestamp}_{chunk_index:04d}.wav"
    save_wav_file(chunk_filename, chunk)
    return chunk_filename


def main():
    """
    Stream non-silent audio from microphone and save each chunk to individual WAV files.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chunk_index = 0
    saved_files: List[str] = []

    for chunk in stream_non_silent_audio(
        silence_threshold=None,
        chunk_duration=0.5,
        silence_duration=2.0
    ):
        chunk_filename = save_chunk(chunk, chunk_index, timestamp)
        saved_files.append(chunk_filename)
        logger.success(f"Saved chunk {chunk_index} to {chunk_filename}")
        chunk_index += 1

    if not saved_files:
        logger.warning("No non-silent audio chunks captured.")
        return

    logger.info(
        f"Streamed audio saved to {len(saved_files)} chunk files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
