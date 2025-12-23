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

            # Helper: Bilingual SRT using proportional split (no true segment timings)
            def segments_to_bilingual_srt(segments: list, full_translation: str) -> str:
                lines = []
                translated_parts = full_translation.split()
                total_words = len(translated_parts)
                seg_count = len(segments) if segments else 1
                words_per_seg = max(1, total_words // seg_count) if seg_count > 0 else 1

                for s_idx, seg in enumerate(segments, start=1):
                    # Fallback: Use 0:00:00,000 → 0:00:00,000 since we have no timings
                    start_tc = "00:00:00,000"
                    end_tc = "00:00:00,000"
                    start_word = (s_idx - 1) * words_per_seg
                    end_word = s_idx * words_per_seg if s_idx < seg_count else total_words
                    seg_translation = " ".join(translated_parts[start_word:end_word])
                    lines.append(str(s_idx))
                    lines.append(f"{start_tc} --> {end_tc}")
                    lines.append(seg)
                    lines.append(seg_translation.strip() if seg_translation else "[No translation]")
                    lines.append("")
                return "\n".join(lines)

            # As we have no segment/timestamp info from the remote server, make "segments" one segment = full ja_text
            segments = [ja_text] if ja_text else []
            bilingual_srt_content = segments_to_bilingual_srt(segments, en_text)

            # Bilingual TXT: Just the two blocks
            bilingual_txt_lines = []
            if ja_text or en_text:
                bilingual_txt_lines.extend([
                    "1",
                    "00:00:00,000 --> 00:00:00,000",
                    ja_text,
                    en_text if en_text else "[No translation]",
                    ""
                ])
            bilingual_txt_content = "\n".join(bilingual_txt_lines)

            # Save per-file artifacts
            save_file(bilingual_srt_content, file_output_dir / "bilingual_subtitles.srt")
            save_file(bilingual_txt_content, file_output_dir / "translation.txt")
            save_file(result, file_output_dir / "info.json")
            save_file([], file_output_dir / "segments.json")  # Placeholder

            all_subtitles_srt_path = base_output / "all_subtitles.srt"
            all_translations_txt_path = base_output / "all_translations.txt"

            file_header = f"\n\n=== FILE: {audio_path.name} ===\n\n"

            with open(all_subtitles_srt_path, "a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write(bilingual_srt_content + "\n")
                else:
                    f.write(file_header + bilingual_srt_content + "\n")

            with open(all_translations_txt_path, "a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write(bilingual_txt_content + "\n")
                else:
                    f.write(file_header + bilingual_txt_content + "\n")

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