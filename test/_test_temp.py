from rich import print as rprint
from rich.table import Table
from rich.console import Console

from pathlib import Path
import json
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

console = Console()

# this assumes that you have a relevant version of PyTorch installed
# !pip install -q torchaudio

SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

USE_PIP = True # download model using pip package or torch.hub
USE_ONNX = False # change this to True if you want to test onnx model
# if USE_ONNX:
#     !pip install -q onnxruntime
if USE_PIP:
    # !pip install -q silero-vad
    from silero_vad import (load_silero_vad,
                            read_audio,
                            get_speech_timestamps,
                            save_audio,
                            VADIterator,
                            collect_chunks)
    model = load_silero_vad(onnx=USE_ONNX)
else:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=USE_ONNX)

    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils

audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20260108_100755.wav"
wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)

# Define output directory (based on input file name + timestamp)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = OUTPUT_DIR / f"segments_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

rprint(f"[bold green]Output directory:[/bold green] {output_dir}")

# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)

# Convert sample indices to seconds for readable output
readable_timestamps = [
    {
        "start": round(start / SAMPLING_RATE, 2),
        "end": round(end / SAMPLING_RATE, 2),
        "duration": round((end - start) / SAMPLING_RATE, 2),
    }
    for timestamp in speech_timestamps
    for start, end in [(timestamp["start"], timestamp["end"])]
]

# Improved rich output
rprint("\n[bold cyan]Speech Detection Results[/bold cyan]\n")

if not readable_timestamps:
    rprint("[yellow]No speech segments detected in the audio.[/yellow]")
else:
    table = Table(show_header=True, header_style="bold magenta", border_style="bright_blue")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration (s)", justify="right")

    for seg in readable_timestamps:
        table.add_row(
            f"{seg['start']:.2f}",
            f"{seg['end']:.2f}",
            f"{seg['duration']:.2f}",
        )

    console.print(table)
    rprint(f"\n[green]Found {len(readable_timestamps)} speech segment(s).[/green]\n")

# === Save audio segments and metadata ===
if speech_timestamps:
    rprint("[bold cyan]Saving speech segments...[/bold cyan]")

    metadata = []

    for idx, ts in enumerate(speech_timestamps, start=1):
        # Generate filename
        segment_filename = f"segment_{idx:03d}.wav"
        segment_path = output_dir / segment_filename

        # Extract and save audio chunk
        chunk = collect_chunks([ts], wav)
        save_audio(segment_path, chunk, sampling_rate=SAMPLING_RATE)

        # Compute times in seconds
        start_s = round(ts["start"] / SAMPLING_RATE, 2)
        end_s = round(ts["end"] / SAMPLING_RATE, 2)
        duration_s = round((ts["end"] - ts["start"]) / SAMPLING_RATE, 2)

        # Add to metadata
        metadata.append({
            "index": idx,
            "start_seconds": start_s,
            "end_seconds": end_s,
            "duration_seconds": duration_s,
            "filename": segment_filename,
            "file_path": str(segment_path)
        })

        rprint(f"  [green]✓[/green] Saved: {segment_filename} ({duration_s}s)")

    # Save metadata
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    rprint("\n[bold green]All segments saved![/bold green]")
    rprint(f"   Segments: {len(metadata)}")
    rprint(f"   Metadata: {meta_path}")
else:
    rprint("[yellow]No segments to save.[/yellow]")