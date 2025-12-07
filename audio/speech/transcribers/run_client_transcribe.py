# client_transcribe.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import httpx
import soundfile as sf
import librosa
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TransferSpeedColumn, BarColumn
from rich.table import Table

console = Console()


def get_audio_duration(file_path: Path) -> float:
    """
    Return real audio duration in seconds.
    Tries the fastest methods first → always returns correct value.
    """
    # 1. soundfile – instant for WAV/FLAC/OGG (no decode)
    try:
        return sf.info(file_path).duration
    except Exception:
        pass

    # 2. librosa with metadata-only read (the correct modern way)
    try:
        # sr=None + mono=False → reads only container metadata when possible
        return float(librosa.get_duration(path=str(file_path), sr=None, mono=False))
    except Exception:
        pass

    # 3. Last resort: use ffprobe via audioread (still used internally by librosa)
    import audioread
    with audioread.audio_open(str(file_path)) as f:
        return f.duration

    # If everything fails (should never happen)
    raise RuntimeError("Could not determine audio duration")


async def transcribe_file(
    file_path: str | Path,
    *,
    url: str = "http://shawn-pc.local:8001/transcribe",
    model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "large-v3",
    compute_type: str = "int8",
    device: Literal["cpu", "cuda"] = "cpu",
    task: Literal["transcribe", "translate"] = "transcribe",
) -> dict:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    # THIS IS THE ONLY LINE THAT CHANGED compared to the previous version
    real_duration = get_audio_duration(file_path)

    console.print(
        f"[bold cyan]Transcribing:[/bold cyan] {file_path.name} "
        f"({file_size_mb:.1f} MB, {real_duration:.2f}s)"
    )

    if task == "translate":
        url = url.replace("/transcribe", "/translate")

    params = {
        "model_size": model_size,
        "compute_type": compute_type,
        "device": device,
    }

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "audio/wav")}

        with Progress(
            SpinnerColumn(),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("[green]Uploading & transcribing...", total=file_path.stat().st_size)

            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, files=files, params=params)
                progress.update(task_id, completed=file_path.stat().st_size)

        response.raise_for_status()
        result = response.json()

    # Result table – now shows the REAL duration
    table = Table(title="Transcription Result", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    transcription = result.get("transcription") or result.get("text", "")
    language = result.get("detected_language") or result.get("language") or "N/A"
    lang_prob = result.get("detected_language_prob", result.get("language_probability", 0.0))

    table.add_row("Text", transcription[:500] + ("..." if len(transcription) > 500 else ""))
    table.add_row("Language", f"{language} ({lang_prob:.2f})")
    table.add_row("Duration", f"[bold green]{real_duration:.2f}s[/bold green]")
    table.add_row("Segments", str(len(result.get("segments", []))) if result.get("segments") else "0")

    if "duration_sec" in result:
        table.add_row("Inference Time", f"{result['duration_sec']:.2f}s")

    console.print(table)
    return result


if __name__ == "__main__":
    example_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps/002_10.00_16.52_SPEAKER_01/combined_sound.wav"

    asyncio.run(
        transcribe_file(
            example_file,
            model_size="small",
            compute_type="int8_float16",
            device="cuda",
            task="transcribe",
        )
    )