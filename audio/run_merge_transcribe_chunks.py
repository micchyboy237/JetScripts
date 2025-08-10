import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf
from typing import List, Optional
from jet.audio.record_mic import SAMPLE_RATE, CHANNELS, DTYPE
from jet.audio.audio_file_transcriber import AudioFileTranscriber
from jet.logger import logger


def merge_audio_chunks(chunk_files: List[str], overlap_duration: float) -> Optional[np.ndarray]:
    """
    Merge audio chunk files into a single audio array, handling overlaps.

    Args:
        chunk_files: List of paths to audio chunk files (WAV format).
        overlap_duration: Duration of overlap between chunks in seconds.

    Returns:
        np.ndarray: Merged audio data, or None if no valid chunks are found.
    """
    if not chunk_files:
        logger.warning("No chunk files provided for merging")
        return None

    overlap_samples = int(SAMPLE_RATE * overlap_duration)
    merged_audio = []

    for i, chunk_file in enumerate(chunk_files):
        try:
            audio_data, file_sample_rate = sf.read(chunk_file)
            if file_sample_rate != SAMPLE_RATE:
                logger.error(
                    f"Sample rate mismatch in {chunk_file}: expected {SAMPLE_RATE}, got {file_sample_rate}")
                return None
            if audio_data.ndim > 1:
                # Convert to mono if stereo
                audio_data = np.mean(audio_data, axis=1)
            if i < len(chunk_files) - 1:  # Trim overlap for all but the last chunk
                audio_data = audio_data[:-
                                        overlap_samples] if overlap_samples > 0 else audio_data
            merged_audio.append(audio_data)
        except Exception as e:
            logger.error(f"Error reading chunk {chunk_file}: {str(e)}")
            return None

    if not merged_audio:
        logger.warning("No valid audio data after processing chunks")
        return None

    merged_audio = np.concatenate(merged_audio, axis=0)
    logger.info(
        f"Merged {len(chunk_files)} chunks into audio of {len(merged_audio)/SAMPLE_RATE:.2f}s")
    return merged_audio


def save_merged_audio(audio_data: np.ndarray, output_path: str):
    """
    Save merged audio data to a WAV file.

    Args:
        audio_data: Audio data as numpy array.
        output_path: Path to save the WAV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_data, SAMPLE_RATE)
    logger.info(f"Merged audio saved to {output_path}")


async def transcribe_merged_audio(audio_path: str, output_dir: str, transcriber: AudioFileTranscriber) -> Optional[str]:
    """
    Transcribe the merged audio file.

    Args:
        audio_path: Path to the merged audio file.
        output_dir: Directory to save the transcription.
        transcriber: AudioFileTranscriber instance.

    Returns:
        Optional[str]: Transcription text, or None if transcription fails.
    """
    transcription = await transcriber.transcribe_from_file(audio_path, output_dir=output_dir)
    if transcription:
        logger.info(
            f"Transcription completed: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
    else:
        logger.warning("No transcription produced from merged audio")
    return transcription


def main():
    """
    Merge audio chunks and transcribe them as a single audio file.
    """
    parser = argparse.ArgumentParser(
        description="Merge and transcribe audio chunks from stream_mic.py."
    )
    parser.add_argument("chunk_dir", type=str,
                        help="Directory containing audio chunk files (e.g., generated/run_stream_mic)")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save merged audio and transcription")
    parser.add_argument("--overlap-duration", type=float, default=1.0,
                        help="Overlap duration between chunks in seconds (default: 1.0)")
    parser.add_argument("--model-size", type=str, default="small",
                        help="Whisper model size for transcription (default: small)")
    args = parser.parse_args()

    # Find all chunk files with consistent timestamp
    chunk_dir = Path(args.chunk_dir)
    chunk_files = sorted([str(f)
                         for f in chunk_dir.glob("stream_chunk_*.wav")])
    if not chunk_files:
        logger.error(f"No chunk files found in {chunk_dir}")
        return

    # Merge chunks
    merged_audio = merge_audio_chunks(chunk_files, args.overlap_duration)
    if merged_audio is None:
        logger.error("Failed to merge audio chunks")
        return

    # Save merged audio
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_audio_path = os.path.join(
        args.output_dir, f"merged_audio_{timestamp}.wav")
    save_merged_audio(merged_audio, merged_audio_path)

    # Transcribe merged audio
    transcriber = AudioFileTranscriber(
        model_size=args.model_size, sample_rate=SAMPLE_RATE)
    transcription = asyncio.run(transcribe_merged_audio(
        merged_audio_path, args.output_dir, transcriber))
    if transcription:
        print(f"Transcription: {transcription}")
    else:
        print("No transcription produced.")


if __name__ == "__main__":
    main()
