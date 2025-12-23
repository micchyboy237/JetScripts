from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Generator, Union

from rich.progress import Progress, SpinnerColumn, TextColumn
from jet.audio.transcribers.base_client_async import (
    atranscribe_audio as transcribe_audio,
    TranscribeResponse,
    aclose_async_client,
)
from jet.audio.utils import AudioInput, resolve_audio_paths
from jet.file.utils import save_file
from jet.logger import logger

from jet.audio.speech.output_utils import (
    _seconds_to_timestamp,
    write_srt_file,
    append_to_combined_srt,
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

async def translate_audio_files(
    audio_inputs: AudioInput,
    *,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    recursive: bool = True,
) -> Generator[Path, None, None]:
    """
    Translate Japanese audio to English, yielding one output directory per file as soon as it is fully processed.

    This allows immediate user feedback and logging per file without waiting for all files to complete.

    Returns:
        Generator yielding Path objects (one per processed file's output directory).
    """
    audio_paths = resolve_audio_paths(audio_inputs, recursive=recursive)

    base_output = Path(output_dir)
    shutil.rmtree(base_output, ignore_errors=True)
    base_output.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing {task.completed}/{task.total} files..."),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Translating", total=len(audio_paths))

        async def process_single_file(audio_path: Path, idx: int) -> Path:
            file_output_dir = base_output / f"translated_{idx:03d}"
            file_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Translating → {audio_path.name}")

            result: TranscribeResponse = await transcribe_audio(audio_path)
            duration = result["duration_sec"]
            ja_text = result["transcription"].strip()
            en_text = result["translation"].strip()
            detected_lang = result["detected_language"]
            detected_prob = result["detected_language_prob"]

            logger.info(f"Duration: {duration:.2f}s | Language: {detected_lang} ({detected_prob:.2%})")

            # Create a single-entry bilingual SRT using reusable utilities (no real timestamps available)
            srt_path = file_output_dir / "bilingual_subtitles.srt"
            write_srt_file(
                filepath=srt_path,
                source_text=ja_text.strip() if ja_text else "",
                target_text=en_text.strip() if en_text else "",
                start_sample=0.0,
                end_sample=duration,
                index=1,
            )

            # Simple bilingual TXT (kept as-is since no equivalent reusable helper exists yet)
            txt_lines = []
            if ja_text or en_text:
                txt_lines.extend([
                    "1",
                    f"{_seconds_to_timestamp(0.0)} --> {_seconds_to_timestamp(duration)}",
                    ja_text.strip() if ja_text else "",
                    en_text.strip() if en_text else "[No translation]",
                    ""
                ])
            txt_content = "\n".join(txt_lines)

            save_file(txt_content, file_output_dir / "translation.txt")
            save_file(result, file_output_dir / "info.json")
            save_file([], file_output_dir / "segments.json")  # Placeholder

            all_subtitles_srt_path = base_output / "all_subtitles.srt"
            all_translations_txt_path = base_output / "all_translations.txt"

            # Append to combined SRT using reusable helper (adds proper single entry)
            current_index = 1  # Single entry per file; combined file will auto-increment via append logic
            if all_subtitles_srt_path.exists():
                # Count existing entries to continue indexing
                existing_content = all_subtitles_srt_path.read_text(encoding="utf-8")
                current_index = len([line for line in existing_content.splitlines() if line.strip().isdigit()]) + 1

            # Add file separator header as plain text before the subtitle block
            file_header = f"\n\n=== FILE: {audio_path.name} ===\n\n"
            if all_subtitles_srt_path.exists():
                all_subtitles_srt_path.write_text(
                    all_subtitles_srt_path.read_text(encoding="utf-8") + file_header,
                    encoding="utf-8",
                )

            append_to_combined_srt(
                combined_path=all_subtitles_srt_path,
                source_text=ja_text.strip() if ja_text else "",
                target_text=en_text.strip() if en_text else "",
                start_sample=0.0,
                end_sample=duration,
                index=current_index,
            )

            # Append simple TXT version with same header
            if all_translations_txt_path.exists():
                all_translations_txt_path.write_text(
                    all_translations_txt_path.read_text(encoding="utf-8") + file_header,
                    encoding="utf-8",
                )
            with open(all_translations_txt_path, "a", encoding="utf-8") as f:
                f.write(txt_content + "\n")

            logger.success(f"Saved → {file_output_dir.name}")

            progress.advance(task_id)

            return file_output_dir

        # Process files sequentially with async transcription calls
        for idx, audio_path in enumerate(audio_paths, start=1):
            audio_path = Path(audio_path)
            file_output_dir = await process_single_file(audio_path, idx)
            yield file_output_dir

    await aclose_async_client()
    logger.success(f"All done! Outputs in:\n   → {base_output.resolve()}")


async def _run_examples() -> None:
    example_files = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps/segments",

        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/sppech_utils/generated/run_check_speech_waves/segments",

        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_analyze_speech/segments",
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_analyze_speech/raw_segments",
    ]

    for file_or_dir in example_files:
        sub_dir = f"{OUTPUT_DIR}/{os.path.basename(file_or_dir)}"
        async for output_dir in translate_audio_files(
            audio_inputs=file_or_dir,
            recursive=True,
            output_dir=sub_dir,
        ):
            print(f"Finished: {output_dir.resolve()}")

if __name__ == "__main__":
    asyncio.run(_run_examples())