import os
import json
from datetime import datetime
import shutil
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from jet.audio.record_mic import save_wav_file, SAMPLE_RATE, detect_silence, calibrate_silence_threshold
from jet.audio.stream_mic import stream_non_silent_audio
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def trim_silent_portions(chunk: np.ndarray, silence_threshold: float, sub_chunk_duration: float = 0.1) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Trim silent portions from the start and end of an audio chunk, preserving overlap.
    Args:
        chunk: Audio chunk to trim (numpy array, including overlap).
        silence_threshold: Energy threshold for silence detection.
        sub_chunk_duration: Duration of sub-chunks for silence detection (seconds).
    Returns:
        Tuple of (trimmed chunk or None if all silent, start index, end index).
    """
    sub_chunk_size = int(SAMPLE_RATE * sub_chunk_duration)
    if chunk.shape[0] < sub_chunk_size:
        logger.debug(f"Chunk too small for trimming: {chunk.shape[0]} samples")
        return chunk, 0, chunk.shape[0] if not detect_silence(chunk, silence_threshold) else (None, 0, 0)

    # Split chunk into sub-chunks
    sub_chunks = [chunk[i:i + sub_chunk_size]
                  for i in range(0, chunk.shape[0], sub_chunk_size)]
    start_idx = 0
    end_idx = len(chunk)

    # Find first non-silent sub-chunk
    for i, sub_chunk in enumerate(sub_chunks):
        if len(sub_chunk) >= sub_chunk_size and not detect_silence(sub_chunk, silence_threshold):
            start_idx = i * sub_chunk_size
            break
    else:
        logger.debug("All sub-chunks are silent")
        return None, 0, 0

    # Find last non-silent sub-chunk
    for i in range(len(sub_chunks) - 1, -1, -1):
        if len(sub_chunks[i]) >= sub_chunk_size and not detect_silence(sub_chunks[i], silence_threshold):
            end_idx = (i + 1) * sub_chunk_size
            break

    trimmed_chunk = chunk[start_idx:end_idx]
    logger.debug(
        f"Trimmed {start_idx} samples from start, {chunk.shape[0] - end_idx} from end, remaining: {len(trimmed_chunk)}")
    return trimmed_chunk, start_idx, end_idx


def save_chunk(chunk: np.ndarray, chunk_index: int, timestamp: str, cumulative_duration: float, silence_threshold: float, overlap_samples: int) -> Tuple[Optional[str], Optional[Dict]]:
    """Save a trimmed audio chunk to a WAV file, including overlap, and return metadata."""
    trimmed_chunk, start_idx, end_idx = trim_silent_portions(
        chunk, silence_threshold)
    if trimmed_chunk is None or len(trimmed_chunk) == 0:
        logger.debug(
            f"Chunk {chunk_index} is entirely silent after trimming, not saved")
        return None, None

    chunk_filename = f"{OUTPUT_DIR}/stream_chunk_{timestamp}_{chunk_index:04d}.wav"
    save_wav_file(chunk_filename, trimmed_chunk)
    chunk_duration = len(trimmed_chunk) / SAMPLE_RATE
    logger.debug(
        f"Saved chunk {chunk_index} to {chunk_filename}, size: {len(trimmed_chunk)} samples, duration: {chunk_duration:.2f}s, "
        f"overlap: {overlap_samples if chunk_index > 0 else 0} samples")
    metadata = {
        "chunk_index": chunk_index,
        "filename": chunk_filename,
        "duration_s": round(chunk_duration, 3),
        "timestamp": timestamp,
        "sample_count": len(trimmed_chunk),
        "start_time_s": round(cumulative_duration, 3),
        "end_time_s": round(cumulative_duration + chunk_duration, 3),
        "trimmed_samples_start": start_idx,
        "trimmed_samples_end": chunk.shape[0] - end_idx,
        "overlap_samples": overlap_samples if chunk_index > 0 else 0
    }
    return chunk_filename, metadata


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
            chunk, chunk_index, timestamp, cumulative_duration, silence_threshold, overlap_samples)
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
