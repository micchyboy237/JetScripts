import asyncio
import os
import json
from pathlib import Path
import shutil
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelsType
import numpy as np
from typing import List, Dict
from jet.audio.transcribers.audio_file_transcriber import AudioFileTranscriber
from jet.audio.transcribers.audio_context_transcriber import AudioContextTranscriber
from jet.audio.record_mic import save_wav_file, SAMPLE_RATE, calibrate_silence_threshold
from jet.audio.stream_mic import async_stream_non_silent_audio, save_chunk
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def transcription_worker(
    queue: asyncio.Queue,
    output_dir: str,
    transcriber_context: AudioContextTranscriber,
    chunks_metadata: list,
    metadata_lock: asyncio.Lock,
    accumulated_transcription_ref: list,
    prev_chunk_filename_ref: list
):
    """Background task that processes transcription jobs from queue."""
    while True:
        chunk_index, chunk_filename = await queue.get()
        if chunk_filename is None:
            queue.task_done()
            break

        prev_file_path = prev_chunk_filename_ref[0]
        next_file_path = None  # Not available yet — future chunks not known

        try:
            # ← NOW FULLY NON-BLOCKING: Whisper runs in ThreadPoolExecutor
            non_overlap_transcription, start_overlap_transcription, _ = await asyncio.get_running_loop().run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                transcriber_context.transcribe_with_context,
                chunk_filename,
                prev_file_path,
                next_file_path,
                2.0 if chunk_index > 0 else 0.0,   # start_overlap_duration
                2.0,                                # end_overlap_duration
                f"{output_dir}/transcriptions"
            )

            async with metadata_lock:
                metadata = next(m for m in chunks_metadata if m["filename"] == chunk_filename)
                metadata["transcription"] = non_overlap_transcription or ""
                if non_overlap_transcription:
                    accumulated_transcription_ref[0] = (
                        accumulated_transcription_ref[0] + " " + non_overlap_transcription
                    ).strip()
                metadata["accumulated_transcription"] = accumulated_transcription_ref[0]
                print(f"Transcribed chunk {chunk_index}: {non_overlap_transcription[:80] if non_overlap_transcription else 'None'}...")

            prev_chunk_filename_ref[0] = chunk_filename

        except Exception as e:
            logger.error(f"Transcription failed for {chunk_filename}: {e}")
        finally:
            queue.task_done()

async def main():
    chunk_index = 0
    saved_files: List[str] = []
    chunks_metadata: List[Dict] = []
    total_samples = 0
    cumulative_duration = 0.0
    min_chunk_duration = 5.0
    overlap_duration = 2.0
    overlap_samples = int(SAMPLE_RATE * overlap_duration)
    all_chunks = []
    model_size: WhisperModelsType = "large-v3"
    save_original_stream = False

    silence_threshold = calibrate_silence_threshold()
    # Only keep context transcriber for async
    transcriber = AudioFileTranscriber(model_size=model_size, sample_rate=None)
    transcriber_context = AudioContextTranscriber(model_size=model_size, sample_rate=None)

    # Setup background transcription worker
    transcription_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    metadata_lock = asyncio.Lock()
    accumulated_transcription_ref = [""]   # mutable list of one (acts as reference)
    prev_chunk_filename_ref = [None]       # mutable list of one (acts as reference)

    worker_task = asyncio.create_task(
        transcription_worker(
            transcription_queue,
            OUTPUT_DIR,
            transcriber_context,
            chunks_metadata,
            metadata_lock,
            accumulated_transcription_ref,
            prev_chunk_filename_ref
        )
    )

    try:
        async for chunk in async_stream_non_silent_audio(
            silence_threshold=silence_threshold,
            chunk_duration=0.5,
            silence_duration=5.0,
            min_chunk_duration=min_chunk_duration,
            overlap_duration=overlap_duration,
            auto_close_on_long_silence=False,
        ):
            chunk_duration_val = len(chunk) / SAMPLE_RATE
            expected_min_duration = min_chunk_duration + overlap_duration
            if chunk_duration_val < expected_min_duration:
                logger.warning(
                    f"Chunk {chunk_index} duration {chunk_duration_val:.2f}s is less than minimum {expected_min_duration:.2f}s")
            if chunk.shape[0] % int(SAMPLE_RATE * 0.5) != 0 and chunk.shape[0] < int(SAMPLE_RATE * expected_min_duration):
                logger.warning(
                    f"Chunk {chunk_index} has non-standard size: {chunk.shape[0]} samples")

            all_chunks.append(chunk)
            chunk_filename, metadata = save_chunk(
                chunk, chunk_index, cumulative_duration, silence_threshold, overlap_samples, OUTPUT_DIR
            )

            if chunk_filename and metadata:
                saved_files.append(chunk_filename)
                chunks_metadata.append(metadata)

                total_samples += metadata["sample_count"] - (overlap_samples if chunk_index > 0 else 0)
                cumulative_duration += metadata["duration_s"] - (overlap_duration if chunk_index > 0 else 0)

                print(f"Saved chunk {chunk_index} → {Path(chunk_filename).name}")
                await transcription_queue.put((chunk_index, chunk_filename))
                chunk_index += 1
            else:
                logger.debug(f"Skipped saving chunk {chunk_index} due to silence")
                chunk_index += 1

    finally:
        # Signal worker to stop
        await transcription_queue.put((None, None))
        await transcription_queue.join()
        await worker_task

    # ------------------------------------------------------------------
    # Optional: save and transcribe the complete original stream
    # ------------------------------------------------------------------
    if all_chunks and save_original_stream:
        original_filename = f"{OUTPUT_DIR}/original_stream_{chunk_index:04d}.wav"
        # remove overlaps to get the true continuous stream
        consolidated_chunks = [all_chunks[0]] + [c[overlap_samples:] for c in all_chunks[1:]]
        original_audio = np.concatenate(consolidated_chunks, axis=0)
        save_wav_file(original_filename, original_audio)
        original_duration = len(original_audio) / SAMPLE_RATE
        logger.info(f"Saved original stream to {original_filename}, duration: {original_duration:.2f}s")

        original_transcription = await transcriber.transcribe_from_file(
            original_filename, f"{OUTPUT_DIR}/transcriptions"
        )
        concatenated_transcription = " ".join(
            meta["transcription"] for meta in chunks_metadata if meta.get("transcription")
        ).strip()

        if original_transcription and concatenated_transcription != original_transcription:
            logger.warning(
                "Original transcription differs from concatenated chunk transcriptions. "
                f"Original: '{original_transcription}' | Concatenated: '{concatenated_transcription}'"
            )
    else:
        original_filename = ""
        original_duration = 0.0
        if all_chunks:
            logger.info("Skipping original stream WAV/transcription (save_original_stream=False)")
        else:
            logger.warning("No chunks recorded – nothing to save")
    metadata_file = os.path.join(OUTPUT_DIR, "chunks_info.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            "total_chunks": len(saved_files),
            "total_duration_s": round(total_samples / SAMPLE_RATE, 3),
            "original_wav": {
                "filename": original_filename,
                "duration_s": round(original_duration, 3) if save_original_stream else 0.0,
                "sample_count": len(original_audio) if (all_chunks and save_original_stream) else 0
            },
            "chunks": chunks_metadata
        }, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")
    print(f"Streamed audio saved to {len(saved_files)} chunk files and original stream in {OUTPUT_DIR}, "
          f"total duration: {cumulative_duration:.2f}s (without overlaps), "
          f"final accumulated transcription: {accumulated_transcription_ref[0]}")

if __name__ == "__main__":
    asyncio.run(main())
