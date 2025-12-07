#!/usr/bin/env python3
"""
Simple & fast Japanese → English translation using faster-whisper
- Uses Whisper large-v3 model (best quality for Japanese)
- Outputs translated English text + optional segmented SRT/VTT
- Clean, reusable, and easy to read
"""

import os
from pathlib import Path
import shutil
from typing import Literal

from faster_whisper import WhisperModel
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

# ==================== INPUTS (modify these) ====================
AUDIO_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/data/audio/1.wav"
INPUT_AUDIO_PATH = Path(AUDIO_FILE)   # Your Japanese audio file
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR = Path(OUTPUT_DIR)                 # Where to save results
MODEL_SIZE = "large-v3"                                # Best for Japanese → English
DEVICE = "cpu"                                        # "cuda" or "cpu" (auto-detects if None)
BEAM_SIZE = 5                                          # Higher = better accuracy, slower
LANGUAGE = "ja"                                        # Source language (Japanese)
TASK: Literal["translate", "transcribe"] = "translate"  # Must be "translate" for ja→en
# ================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Loading faster-whisper model:[/bold green] {MODEL_SIZE}")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type="int8_float16" if DEVICE == "cuda" else "int8",
    )

    console.print(f"[bold blue]Transcribing & translating:[/bold blue] {INPUT_AUDIO_PATH.name}")
    segments, info = model.transcribe(
        str(INPUT_AUDIO_PATH),
        language=LANGUAGE,
        task=TASK,
        beam_size=BEAM_SIZE,
        vad_filter=True,                # Removes silence → faster & cleaner
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Show detection info
    table = Table(title="Detection Info")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Detected Language", f"{info.language} ({info.language_probability:.2%})")
    table.add_row("Duration", f"{info.duration:.2f}s")
    console.print(table)

    # Save results
    txt_path = OUTPUT_DIR / f"{INPUT_AUDIO_PATH.stem}_en.txt"
    srt_path = OUTPUT_DIR / f"{INPUT_AUDIO_PATH.stem}_en.srt"

    with open(txt_path, "w", encoding="utf-8") as f_txt, \
         open(srt_path, "w", encoding="utf-8") as f_srt:

        console.print("[bold cyan]Segments:[/bold cyan]")
        for i, segment in enumerate(tqdm(segments, desc="Writing", unit="seg")):
            text = segment.text.strip()

            # Plain text
            f_txt.write(text + "\n")

            # SRT subtitle format
            start = segment.start
            end = segment.end
            f_srt.write(f"{i+1}\n")
            f_srt.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f_srt.write(f"{text}\n\n")

            console.print(f"[{start:.2f}-{end:.2f}] {text}")

    console.print("\n[bold green]Done![/] Translation saved to:")
    console.print(f"   • Text: {txt_path}")
    console.print(f"   • SRT:  {srt_path}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


if __name__ == "__main__":
    main()