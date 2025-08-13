import asyncio
import os
import json
import soundfile as sf
import shutil
import sys
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from jet.audio.audio_file_transcriber import AudioFileTranscriber
from jet.audio.audio_context_transcriber import AudioContextTranscriber
from jet.audio.record_mic import save_wav_file, SAMPLE_RATE, detect_silence, calibrate_silence_threshold
from jet.audio.stream_mic import save_chunk, stream_non_silent_audio
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    """
    Stream non-silent audio from microphone, save trimmed non-silent chunks with overlaps to individual WAV files,
    transcribe each chunk with context, update previous transcriptions for consistency, and save a single original WAV file.
    Save metadata to chunks_info.json with start_time_s, end_time_s, transcription, and accumulated_transcription, rounded to 3 decimal places.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_index = 0
    saved_files: List[str] = []
    chunks_metadata: List[Dict] = []
    total_samples = 0
    cumulative_duration = 0.0
    min_chunk_duration = 1.0
    overlap_duration = 1.0
    overlap_samples = int(SAMPLE_RATE * overlap_duration)
    all_chunks = []
    silence_threshold = calibrate_silence_threshold()
    transcriber = AudioFileTranscriber(model_size="small", sample_rate=None)
    transcriber_context = AudioContextTranscriber(
        model_size="small", sample_rate=None)
    prev_chunk_filename: Optional[str] = None
    accumulated_transcription = ""
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
        all_chunks.append(chunk)
        chunk_filename, metadata = save_chunk(
            chunk, chunk_index, cumulative_duration, silence_threshold, overlap_samples, OUTPUT_DIR)
        if chunk_filename and metadata:
            saved_files.append(chunk_filename)
            next_chunk_filename = saved_files[chunk_index +
                                              1] if chunk_index + 1 < len(saved_files) else None
            non_overlap_transcription, start_overlap_transcription, end_overlap_transcription = await transcriber_context.transcribe_with_context(
                chunk_filename,
                prev_file_path=prev_chunk_filename,
                next_file_path=next_chunk_filename,
                start_overlap_duration=overlap_duration if chunk_index > 0 else 0.0,
                end_overlap_duration=overlap_duration,
                output_dir=f"{OUTPUT_DIR}/transcriptions"
            )
            metadata["transcription"] = non_overlap_transcription if non_overlap_transcription else ""
            if non_overlap_transcription:
                accumulated_transcription = (
                    accumulated_transcription + " " + non_overlap_transcription).strip()
            metadata["accumulated_transcription"] = accumulated_transcription
            if chunk_index > 0 and start_overlap_transcription and chunks_metadata:
                prev_metadata = chunks_metadata[-1]
                prev_end_overlap_samples = prev_metadata["end_overlap_samples"]
                prev_end_overlap_duration = prev_end_overlap_samples / SAMPLE_RATE
                prev_transcription_file = os.path.join(
                    OUTPUT_DIR, f"transcriptions/transcription_{chunk_index - 1:05d}.txt")
                prev_audio, prev_sr = sf.read(prev_metadata["filename"])
                prev_end_overlap = prev_audio[- prev_end_overlap_samples:
                                              ] if prev_end_overlap_samples > 0 else np.array([])
                if len(prev_end_overlap) > 0:
                    temp_filename = f"{OUTPUT_DIR}/temp_prev_end_overlap_{chunk_index - 1:04d}.wav"
                    save_wav_file(temp_filename, prev_end_overlap)
                    prev_end_transcription, _, _ = await transcriber_context.transcribe_with_context(
                        temp_filename,
                        start_overlap_duration=0.0,
                        end_overlap_duration=0.0
                    )
                    os.remove(temp_filename)
                    if prev_end_transcription and prev_end_transcription != start_overlap_transcription:
                        logger.info(
                            f"Updating previous chunk {chunk_index - 1} transcription due to overlap mismatch")
                        prev_non_overlap_samples = prev_metadata["sample_count"] - \
                            prev_end_overlap_samples
                        prev_non_overlap_duration = prev_non_overlap_samples / SAMPLE_RATE
                        prev_segments, _ = transcriber.model.transcribe(
                            prev_audio,
                            language="en",
                            beam_size=5,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        prev_non_overlap_text = []
                        for segment in prev_segments:
                            if segment.end <= prev_non_overlap_duration:
                                prev_non_overlap_text.append(
                                    segment.text.strip())
                            else:
                                break
                        updated_prev_transcription = " ".join(
                            prev_non_overlap_text).strip() + " " + start_overlap_transcription
                        prev_metadata["transcription"] = updated_prev_transcription
                        prev_metadata["accumulated_transcription"] = accumulated_transcription
                        with open(prev_transcription_file, "w", encoding="utf-8") as f:
                            f.write(updated_prev_transcription)
                        logger.debug(
                            f"Updated transcription for chunk {chunk_index - 1}: {updated_prev_transcription}")
            chunks_metadata.append(metadata)
            total_samples += metadata["sample_count"] - \
                (overlap_samples if chunk_index > 0 else 0)
            cumulative_duration += metadata["duration_s"] - \
                (overlap_duration if chunk_index > 0 else 0)
            print(f"Saved chunk {chunk_index} to {chunk_filename}, samples: {metadata['sample_count']}, "
                  f"duration: {metadata['duration_s']:.2f}s, overlap: {overlap_duration:.2f}s, "
                  f"transcription: {non_overlap_transcription if non_overlap_transcription else 'None'}, "
                  f"accumulated: {accumulated_transcription}")
            prev_chunk_filename = chunk_filename
            chunk_index += 1
        else:
            logger.debug(f"Skipped saving chunk {chunk_index} due to silence")
            chunk_index += 1
    if all_chunks:
        original_filename = f"{OUTPUT_DIR}/original_stream_{chunk_index:04d}.wav"
        consolidated_chunks = [all_chunks[0]] + \
            [chunk[overlap_samples:] for chunk in all_chunks[1:]]
        original_audio = np.concatenate(consolidated_chunks, axis=0)
        save_wav_file(original_filename, original_audio)
        original_duration = len(original_audio) / SAMPLE_RATE
        logger.info(
            f"Saved original stream to {original_filename}, duration: {original_duration:.2f}s")
        original_transcription = await transcriber.transcribe_from_file(original_filename, f"{OUTPUT_DIR}/transcriptions")
        concatenated_transcription = " ".join(
            meta["transcription"] for meta in chunks_metadata if meta["transcription"]
        ).strip()
        if original_transcription and concatenated_transcription != original_transcription:
            logger.warning(
                f"Original transcription differs from concatenated chunk transcriptions. "
                f"Original: '{original_transcription}', Concatenated: '{concatenated_transcription}'")
    else:
        original_filename = ""
        original_duration = 0.0
        logger.warning("No chunks to save for original stream WAV")
    metadata_file = os.path.join(OUTPUT_DIR, "chunks_info.json")
    with open(metadata_file, 'w') as f:
        json.dump({
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
          f"total duration: {cumulative_duration:.2f}s (without overlaps), "
          f"final accumulated transcription: {accumulated_transcription}")

if __name__ == "__main__":
    asyncio.run(main())
