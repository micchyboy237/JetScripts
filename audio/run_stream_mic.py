import os
import json
from datetime import datetime
import shutil
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from jet.audio.record_mic import save_wav_file, SAMPLE_RATE, detect_silence, calibrate_silence_threshold
from jet.audio.stream_mic import save_chunk, stream_non_silent_audio
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main():
    """
    Stream non-silent audio from microphone, save trimmed non-silent chunks with overlaps to individual WAV files,
    and save a single original WAV file containing all non-silent chunks with overlaps consolidated.
    Save metadata to chunks_info.json with start_time_s and end_time_s rounded to 3 decimal places.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chunk_index = 0
    saved_files: List[str] = []
    chunks_metadata: List[Dict] = []
    total_samples = 0
    cumulative_duration = 0.0
    min_chunk_duration = 1.0
    overlap_duration = 1.0
    overlap_samples = int(SAMPLE_RATE * overlap_duration)
    all_chunks = []  # Accumulate all non-silent chunks for original WAV
    # Match threshold used in stream_non_silent_audio
    silence_threshold = calibrate_silence_threshold()

    for chunk in stream_non_silent_audio(
        silence_threshold=silence_threshold,
        chunk_duration=0.5,
        silence_duration=5.0,
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

        # Always append chunk for original WAV to preserve full non-silent stream
        all_chunks.append(chunk)

        # Save only non-silent, trimmed chunks as individual files, including overlaps
        chunk_filename, metadata = save_chunk(
            chunk, chunk_index, timestamp, cumulative_duration, silence_threshold, overlap_samples, OUTPUT_DIR)
        if chunk_filename and metadata:
            saved_files.append(chunk_filename)
            chunks_metadata.append(metadata)
            # Adjust for overlap
            total_samples += metadata["sample_count"] - \
                (overlap_samples if chunk_index > 0 else 0)
            cumulative_duration += metadata["duration_s"] - \
                (overlap_duration if chunk_index > 0 else 0)
            print(f"Saved chunk {chunk_index} to {chunk_filename}, samples: {metadata['sample_count']}, "
                  f"duration: {metadata['duration_s']:.2f}s, overlap: {overlap_duration:.2f}s")
            chunk_index += 1
        else:
            logger.debug(f"Skipped saving chunk {chunk_index} due to silence")
            chunk_index += 1

    if not saved_files:
        print("No non-silent audio chunks captured after trimming.")
        # Still save original WAV if any chunks were yielded
        if all_chunks:
            original_filename = f"{OUTPUT_DIR}/original_stream_{timestamp}.wav"
            consolidated_chunks = [
                all_chunks[0]] + [chunk[overlap_samples:] for chunk in all_chunks[1:]]
            original_audio = np.concatenate(consolidated_chunks, axis=0)
            save_wav_file(original_filename, original_audio)
            original_duration = len(original_audio) / SAMPLE_RATE
            logger.info(
                f"Saved original stream to {original_filename}, duration: {original_duration:.2f}s")
        else:
            original_filename = ""
            original_duration = 0.0
            logger.warning("No chunks to save for original stream WAV")
    else:
        # Save original WAV file with overlaps consolidated
        original_filename = f"{OUTPUT_DIR}/original_stream_{timestamp}.wav"
        consolidated_chunks = [all_chunks[0]] + \
            [chunk[overlap_samples:] for chunk in all_chunks[1:]]
        original_audio = np.concatenate(consolidated_chunks, axis=0)
        save_wav_file(original_filename, original_audio)
        original_duration = len(original_audio) / SAMPLE_RATE
        logger.info(
            f"Saved original stream to {original_filename}, duration: {original_duration:.2f}s")

    metadata_file = os.path.join(OUTPUT_DIR, "chunks_info.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_chunks": len(saved_files),
            "total_duration_s": round(total_samples / SAMPLE_RATE, 3),
            "original_wav": {
                "filename": original_filename,
                "duration_s": round(original_duration, 3),
                "sample_count": len(original_audio) if all_chunks else 0
            },
            "chunks": chunks_metadata
        }, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")
    print(f"Streamed audio saved to {len(saved_files)} chunk files and original stream in {OUTPUT_DIR}, "
          f"total duration: {cumulative_duration:.2f}s (without overlaps)")


if __name__ == "__main__":
    main()
