# JetScripts/audio/transcribers/run_transcribe_audio_silero.py
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from jet.audio.transcribers.utils import segments_to_srt
from jet.audio.utils import SegmentGroup, resolve_audio_paths_by_groups
from jet.file.utils import save_file
from jet.logger import logger

console = Console()

# Output directory next to this script
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


TaskType = Literal["transcribe", "translate"]


def transcribe_segment_groups(
    audio_dir: str | Path,
    *,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str = "ja",
    task: TaskType = "translate",           # ← Fixed: must be valid
    output_dir: Path | str = OUTPUT_DIR,
    vad_filter: bool = True,
    word_timestamps: bool = True,
) -> list[Path]:
    """
    Transcribe all audio files in grouped segments (root + strong/weak chunks).
    One clean output folder per segment.
    """
    audio_dir = Path(audio_dir).resolve()
    output_base = Path(output_dir)
    shutil.rmtree(output_base, ignore_errors=True)
    output_base.mkdir(parents=True, exist_ok=True)

    console.log(f"[bold green]Loading Whisper model:[/] {model_name} → {device} ({compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    console.log("[bold green]Model loaded successfully[/]")

    grouped: dict[str, SegmentGroup] = resolve_audio_paths_by_groups(audio_dir)

    total_files = sum(
        (1 if g["root"] else 0) + len(g["strong_chunks"]) + len(g["weak_chunks"])
        for g in grouped.values()
    )

    created_dirs: list[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Transcribing {task.completed}/{task.total} files..."),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Transcribing", total=total_files)

        for segment_name, group in grouped.items():
            segment_output = output_base / segment_name
            segment_output.mkdir(parents=True, exist_ok=True)
            created_dirs.append(segment_output)

            # Show segment summary
            table = Table(title=f"Segment: [bold magenta]{segment_name}[/]", show_header=True)
            table.add_column("Type", style="dim")
            table.add_column("File", style="green")
            if group["root"]:
                table.add_row("root", Path(group["root"]).name)
            for f in group["strong_chunks"]:
                table.add_row("strong", Path(f).name)
            for f in group["weak_chunks"]:
                table.add_row("weak", Path(f).name)
            console.print(table)

            # Collect all audio files
            files_to_process: list[str] = []
            if group["root"]:
                files_to_process.append(group["root"])
            files_to_process.extend(group["strong_chunks"])
            files_to_process.extend(group["weak_chunks"])

            for audio_path_str in files_to_process:
                audio_path = Path(audio_path_str)
                rel_path = audio_path.relative_to(audio_dir)

                console.log(f"[cyan]→ Transcribing:[/] {rel_path}")

                try:
                    segments, info = model.transcribe(
                        audio=audio_path_str,
                        language=language,
                        task=task,                    # ← guaranteed valid
                        vad_filter=vad_filter,
                        word_timestamps=word_timestamps,
                        chunk_length=30,
                        beam_size=5,
                        best_of=5,
                        patience=1.0,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False,
                        initial_prompt=None,
                    )

                    all_segments = list(segments)
                    full_text = " ".join(seg.text.strip() for seg in all_segments)

                    # Output per file
                    file_out = segment_output / audio_path.stem
                    file_out.mkdir(exist_ok=True)

                    (file_out / "translation.txt").write_text(
                        f"Source: {audio_path.name}\n"
                        f"Segment: {segment_name}\n"
                        f"Model: {model_name}\n"
                        f"Task: Japanese → English ({task})\n"
                        f"Processed: {datetime.now().isoformat()}\n"
                        f"Duration: {info.duration:.2f}s\n"
                        f"Segments: {len(all_segments)}\n"
                        f"Language: {info.language} (prob: {info.language_probability:.2f})\n"
                        f"{'='*60}\n"
                        f"FULL TEXT\n"
                        f"{'='*60}\n\n"
                        f"{full_text}\n",
                        encoding="utf-8",
                    )

                    save_file(info, file_out / "info.json")
                    save_file(all_segments, file_out / "segments.json")
                    save_file(segments_to_srt(all_segments), file_out / "subtitles.srt")

                    console.log(f"[green]Saved[/] → {file_out.relative_to(output_base)}")
                    progress.advance(task_id)

                except Exception as e:
                    console.log(f"[red]Failed[/] {rel_path}: {e}")
                    logger.error(f"Transcription failed for {audio_path}: {e}")

    console.log(f"[bold green]All done![/] Outputs saved to:\n  → {output_base.resolve()}")
    return created_dirs


if __name__ == "__main__":
    AUDIO_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/silero/generated/silero_vad_stream"

    transcribe_segment_groups(
        audio_dir=AUDIO_DIR,
        model_name="large-v3",
        device="cpu",           # or "cuda" if available
        compute_type="int8",
        task="translate",       # ← explicitly set
        vad_filter=True,
    )