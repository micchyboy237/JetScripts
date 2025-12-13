# jet/audio/transcribers/faster_whisper_translator.py

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Union

from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

# Reuse remote transcription client instead of local faster-whisper
from jet.audio.transcribers.base_client import transcribe_audio, TranscribeResponse

from jet.audio.utils import AudioInput, resolve_audio_paths
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)

# Simple helper to create a one-block SRT from full text (keeps compatibility)
def full_text_to_srt(text: str, duration: float) -> str:
    def format_srt_time(sec: float) -> str:
        # Format seconds (float) as SRT time (hh:mm:ss,mmm)
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        seconds = int(sec % 60)
        millis = int((sec - int(sec)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
    start = "00:00:00,000"
    end = format_srt_time(duration)
    return f"1\n{start} --> {end}\n{text.strip()}\n"


def translate_audio_files(
    audio_inputs: AudioInput,
    *,
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "ja",
    task: str = "translate",
    output_dir: Union[str, Path] = OUTPUT_DIR,
    vad_filter: bool = False,
    word_timestamps: bool = True,
    chunk_length: int = 30,
    recursive: bool = False,
) -> List[Path]:
    """
    Translate Japanese audio to English.

    Now supports:
      • Single file
      • List of files
      • Directory → scans recursively or non-recursively for audio files

    Args:
        audio_inputs: Path to an audio file, a list of audio file paths, or a directory containing audio files.
        model_name: Name or path of the Whisper model to use (e.g., "large-v3").
        device: Device to run inference on (e.g., "cpu", "cuda", "mps").
        compute_type: Precision type for inference (e.g., "int8", "float16", "float32").
        language: Input audio language code (e.g., "ja" for Japanese, "en" for English, etc.); used for model inference.
        task: Task type for Whisper ("translate" to always output English, or "transcribe" to keep original).
        output_dir: Directory path to write generated transcript files and outputs.
        vad_filter: Whether to use Voice Activity Detection (VAD) to filter non-speech portions.
        word_timestamps: Whether to include word-level timestamps in the output segments.
        chunk_length: Audio chunk length in seconds to process at once.
        recursive: Whether to search directories for audio files recursively.

    Returns:
        List of output directories (one per processed file)
    """
    audio_paths = resolve_audio_paths(audio_inputs, recursive=recursive)

    # Create base output directory
    base_output = Path(output_dir)
    shutil.rmtree(base_output, ignore_errors=True)
    base_output.mkdir(parents=True, exist_ok=True)

    # Model is now remote – no local loading needed
    logger.info("Loading Whisper model (remote API client in use)")
    logger.success("Model loaded")

    created_dirs: List[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing {task.completed}/{task.total} files..."),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Translating", total=len(audio_paths))

        for idx, audio_path in enumerate(tqdm(audio_paths, desc="Translating", unit="file", colour="cyan"), start=1):
            audio_path = Path(audio_path)

            # Subdir as translated_<num>
            file_output_dir = base_output / f"translated_{idx:03d}"
            file_output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(file_output_dir)

            logger.info(f"Translating → {audio_path.name}")

            # ---- Remote transcription call ----
            try:
                result: TranscribeResponse = transcribe_audio(str(audio_path))
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path.name}: {e}")
                progress.advance(task_id)
                continue

            full_text = result["translation"].strip()
            duration = result["duration_sec"]
            detected_lang = result["detected_language"]
            logger.info(
                f"Duration: {duration:.2f}s | Detected: {detected_lang} "
                f"(prob {result['detected_language_prob']:.2f})"
            )

            # Save translation.txt
            txt_path = file_output_dir / "translation.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Source: {audio_path.name}\n"
                    f"Model: Remote Whisper API\n"
                    f"Task: Japanese → English Translation\n"
                    f"Processed: {datetime.now().isoformat()}\n"
                    f"Duration: {duration:.2f}s\n"
                    f"Detected language: {detected_lang}\n"
                    f"{'='*60}\n"
                    f"FULL ENGLISH TRANSLATION\n"
                    f"{'='*60}\n\n"
                    f"{full_text}\n"
                )

            # Minimal SRT with the whole translation as one subtitle
            srt_content = full_text_to_srt(full_text, duration)
            # Save raw API response for debugging / later re-parsing
            save_file(result, file_output_dir / "response.json")
            save_file(srt_content, file_output_dir / "subtitles.srt")

            logger.success(f"Saved → {file_output_dir.name}")
            progress.advance(task_id)

    logger.success(f"All done! Outputs in:\n   → {base_output.resolve()}")
    return created_dirs


# ==============================
# Main Block — Now super flexible
# ==============================
if __name__ == "__main__":
    example_files = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_speech_detection/segments",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/diarize_file/segments",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/stream_speakers_extractor",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/silero/generated/silero_vad_stream",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/stream_speakers_extractor/speakers/004_13.23_16.64_SPEAKER_01/sound.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_speakers/diarized_001_recording_3_speakers/segments",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_stream_device_output",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/data/audio/1.wav",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_audio_preprocessor/recording_3_speakers_20251208_000155_299/preprocessed.wav",
    ]

    translate_audio_files(
        audio_inputs=example_files,
        recursive=True,
    )