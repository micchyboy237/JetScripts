import asyncio
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as1
from typing import List, Dict, Optional
import soundfile as sf
from jet.audio.record_mic import SAMPLE_RATE, CHANNELS, DTYPE
from jet.audio.audio_file_transcriber import AudioFileTranscriber
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)


async def transcribe_audio_segments(
    audio_path: str,
    chunks_metadata: List[Dict],
    output_dir: str,
    transcriber: AudioFileTranscriber
) -> List[Dict]:
    """
    Transcribe non-silent segments of the audio file based on chunk metadata.
    Args:
        audio_path: Path to the merged audio file.
        chunks_metadata: List of chunk metadata from chunks_info.json.
        output_dir: Directory to save transcription output.
        transcriber: AudioFileTranscriber instance.
    Returns:
        List of dictionaries containing chunk info and transcription text.
    """
    audio_data, file_sample_rate = sf.read(audio_path)
    if file_sample_rate != SAMPLE_RATE:
        logger.error(
            f"Sample rate mismatch in {audio_path}: expected {SAMPLE_RATE}, got {file_sample_rate}")
        return []

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    transcriptions = []
    for chunk_info in chunks_metadata:
        start_sample = int(chunk_info["start_time_s"] * SAMPLE_RATE)
        end_sample = int(chunk_info["end_time_s"] * SAMPLE_RATE)
        if end_sample > len(audio_data):
            logger.warning(
                f"Chunk {chunk_info['chunk_index']} end sample {end_sample} exceeds audio length {len(audio_data)}")
            end_sample = len(audio_data)

        segment = audio_data[start_sample:end_sample]
        if len(segment) == 0:
            logger.debug(
                f"Empty segment for chunk {chunk_info['chunk_index']}, skipping")
            continue

        temp_audio_path = os.path.join(
            output_dir, f"temp_segment_{chunk_info['chunk_index']:04d}.wav")
        sf.write(temp_audio_path, segment, SAMPLE_RATE)

        transcription = await transcriber.transcribe_from_file(temp_audio_path, output_dir=output_dir)
        os.remove(temp_audio_path)  # Clean up temporary file

        transcription_info = {
            "chunk_index": chunk_info["chunk_index"],
            "filename": chunk_info["filename"],
            "start_time_s": chunk_info["start_time_s"],
            "end_time_s": chunk_info["end_time_s"],
            "duration_s": chunk_info["duration_s"],
            "transcription": transcription if transcription else ""
        }
        if transcription:
            logger.info(
                f"Transcribed chunk {chunk_info['chunk_index']}: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
        else:
            logger.warning(
                f"No transcription for chunk {chunk_info['chunk_index']}")

        transcriptions.append(transcription_info)

    return transcriptions


def save_transcription_metadata(transcriptions: List[Dict], output_dir: str, timestamp: str, total_duration: float):
    """
    Save transcription metadata to a JSON file.
    Args:
        transcriptions: List of transcription info dictionaries.
        output_dir: Directory to save the transcription.json file.
        timestamp: Timestamp for the output file.
        total_duration: Total duration of the audio in seconds.
    """
    metadata = {
        "timestamp": timestamp,
        "total_chunks": len(transcriptions),
        "total_duration_s": round(total_duration, 3),
        "transcriptions": transcriptions
    }
    output_path = os.path.join(output_dir, f"transcription_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved transcription metadata to {output_path}")


def main():
    """
    Transcribe non-silent segments of the original merged audio file using chunk metadata and save transcription metadata.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe non-silent segments of the original merged audio file."
    )
    parser.add_argument("chunk_dir", type=str,
                        help="Directory containing chunks_info.json and original_stream_*.wav (e.g., generated/run_stream_mic)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save transcription output (default: {OUTPUT_DIR})")
    parser.add_argument("--model-size", type=str, default="small",
                        help="Whisper model size for transcription (default: small)")
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    metadata_file = chunk_dir / "chunks_info.json"
    if not metadata_file.exists():
        logger.error(f"Metadata file {metadata_file} not found")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    original_audio_path = metadata["original_wav"]["filename"]
    if not Path(original_audio_path).exists():
        logger.error(f"Original audio file {original_audio_path} not found")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    transcriber = AudioFileTranscriber(
        model_size=args.model_size, sample_rate=SAMPLE_RATE)
    transcriptions = asyncio.run(transcribe_audio_segments(
        original_audio_path, metadata["chunks"], args.output_dir, transcriber
    ))

    if not transcriptions:
        logger.warning("No transcriptions produced")
        print("No transcriptions produced.")
        return

    save_transcription_metadata(
        transcriptions, args.output_dir, timestamp, metadata["total_duration_s"])
    for t in transcriptions:
        print(
            f"Chunk {t['chunk_index']}: {t['transcription'][:50]}{'...' if len(t['transcription']) > 50 else ''}")


if __name__ == "__main__":
    main()
