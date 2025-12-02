# jet/audio/transcribers/faster_whisper_translator.py

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Union

from faster_whisper import WhisperModel
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from jet.audio.transcribers.utils import segments_to_srt
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

# Supported audio extensions (common + comprehensive)
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm", ".mp4", ".mkv", ".avi"
}

AudioInput = Union[str, Path, Sequence[Union[str, Path]]]


def _resolve_audio_paths(audio_inputs: AudioInput) -> List[Path]:
    """Resolve single/list/dir → flat list of existing audio file Paths (non-recursive)."""
    inputs = [audio_inputs] if isinstance(audio_inputs, (str, Path)) else audio_inputs
    resolved: List[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_dir():
            # Non-recursive: only direct children
            audio_files = [
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
            ]
            if not audio_files:
                logger.warning(f"No audio files found in directory: {path}")
            resolved.extend(audio_files)
        elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            resolved.append(path)
        elif path.exists():
            logger.warning(f"Skipping non-audio or unsupported file: {path}")
        else:
            logger.error(f"Path not found: {path}")

    if not resolved:
        raise ValueError("No valid audio files found from provided inputs.")

    return resolved


def translate_audio_files(
    audio_inputs: AudioInput,
    *,
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "ja",
    task: str = "translate",
    output_dir: Union[str, Path] = OUTPUT_DIR,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    chunk_length: int = 30,
) -> List[Path]:
    """
    Translate Japanese audio to English.

    Now supports:
      • Single file
      • List of files
      • Directory → scans non-recursively for audio files

    Args:
        audio_inputs: File path, list of paths, or directory
        ... (other args unchanged)

    Returns:
        List of output directories (one per processed file)
    """
    audio_paths = _resolve_audio_paths(audio_inputs)

    # Create base output directory
    base_output = Path(output_dir)
    shutil.rmtree(base_output, ignore_errors=True)
    base_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Whisper model '{model_name}' on {device} ({compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    logger.success("Model loaded")

    created_dirs: List[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing {task.completed}/{task.total} files..."),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Translating", total=len(audio_paths))

        for audio_path in tqdm(audio_paths, desc="Translating", unit="file", colour="cyan"):
            stem = audio_path.stem
            file_output_dir = base_output / f"{stem}_translated"
            file_output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(file_output_dir)

            logger.info(f"Translating → {audio_path.name}")

            segments, info = model.transcribe(
                audio=str(audio_path),
                language=language,
                task=task,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                chunk_length=chunk_length,
                without_timestamps=False,
                condition_on_previous_text=True,
                log_progress=True
            )

            all_segments = list(segments)
            full_text = " ".join(seg.text.strip() for seg in all_segments)

            logger.info(f"Duration: {info.duration:.2f}s | Segments: {len(all_segments)}")

            # Save full translation
            txt_path = file_output_dir / "translation.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Source: {audio_path.name}\n"
                    f"Model: {model_name}\n"
                    f"Task: Japanese → English Translation\n"
                    f"Processed: {datetime.now().isoformat()}\n"
                    f"Duration: {info.duration:.2f}s\n"
                    f"Segments: {len(all_segments)}\n"
                    f"{'='*60}\n"
                    f"FULL ENGLISH TRANSLATION\n"
                    f"{'='*60}\n\n"
                    f"{full_text}\n"
                )

            # Save artifacts
            srt_content = segments_to_srt(all_segments)
            save_file(info, file_output_dir / "info.json")
            save_file(all_segments, file_output_dir / "segments.json")
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
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_streaming_speech_analyzer",
        # Add more files here for batch processing
    ]

    translate_audio_files(
        audio_inputs=example_files,
        model_name="large-v3",
        device="cpu",
        compute_type="int8",
    )