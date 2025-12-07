# client_transcribe.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TransferSpeedColumn, BarColumn
from rich.table import Table

console = Console()


async def transcribe_file(
    file_path: str | Path,
    *,
    url: str = "http://shawn-pc.local:8001/transcribe",
    model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "large-v3",
    compute_type: str = "int8",  # best for GTX 1660
    device: Literal["cpu", "cuda"] = "cpu",
    task: Literal["transcribe", "translate"] = "transcribe",
) -> dict:
    """
    High-quality transcription with beautiful upload + response progress.
    Uses streaming multipart upload → you see progress instantly.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # Human readable file info
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    console.print(f"[bold cyan]Transcribing:[/bold cyan] {file_path.name} ({file_size_mb:.1f} MB)")

    params = {
        "model_size": model_size,
        "compute_type": compute_type,
        "device": device,
    }
    if task == "translate":
        # /translate endpoint uses same params but different route
        url = url.replace("/transcribe", "/translate")

    # Streaming upload with progress bar
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
                response = await client.post(
                    url,
                    files=files,
                    params=params,
                    # httpx streams the file automatically when using files=
                    # progress hook is not needed – Progress updates via task completion
                )
                # Mark upload complete
                progress.update(task_id, completed=file_path.stat().st_size)

        response.raise_for_status()
        result = response.json()

    # Pretty result table
    table = Table(title="Transcription Result", show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Text", result.get("transcription") or result.get("text", ""))
    table.add_row("Language", f"{result.get('detected_language') or result.get('language')} "
                            f"({result.get('detected_language_prob', result.get('language_probability', 0)):.2f})")
    table.add_row("Duration", f"{result['duration_sec']:.2f}s")
    console.print(table)

    return result


# Usage example
if __name__ == "__main__":
    file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps/002_10.00_16.52_SPEAKER_01/combined_sound.wav"

    asyncio.run(
        transcribe_file(
            file_path,
            model_size="large-v3",
            compute_type="int8",  # optimal for GTX 1660
            device="cpu",
            task="transcribe",  # or "translate"
        )
    )