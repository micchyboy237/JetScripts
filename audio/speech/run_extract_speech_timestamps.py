from collections import defaultdict
import json
from typing import Dict, Any

from pathlib import Path
from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps
from jet.file.utils import save_file
from rich.console import Console
import torchaudio
from silero_vad.utils_vad import read_audio
import os
import shutil
from torch import cat  # ── Add this import at the top (if not already present) ──

console = Console()

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def create_sub_dir(file: str):
    pass

def main(audio_file: str | Path, output_dir: str | Path):
    audio_file = str(audio_file)
    output_dir = Path(output_dir)

    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")

    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.3,
        sampling_rate=16000,
        return_seconds=True,
        time_resolution=2,
    )

    waveform = read_audio(audio_file, sampling_rate=16000).unsqueeze(0)  # (1, samples)

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments or [])}\n")

    # ── NEW: Early exit if no segments ───────────────────────────────
    if not segments:
        console.print("[bold yellow]No speech segments detected – skipping save.[/bold yellow]")
        return

    # Only now create the output directory (and subdir logic if needed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
        )

        folder_name = (
            f"{seg['idx']:03d}_"
            f"{seg['start']:.2f}_"
            f"{seg['end']:.2f}_"
            f"{seg['duration']:.3f}_"
            f"{seg['prob']:.3f}"
        )
        seg_dir = output_dir / folder_name
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        save_file(seg, seg_dir / "segment.json")

        # Save audio slice
        start_sample = int(seg['start'] * 16000)
        end_sample = int(seg['end'] * 16000)
        segment_audio = waveform[:, start_sample:end_sample]

        torchaudio.save(
            str(seg_dir / "sound.wav"),
            segment_audio,
            sample_rate=16000,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        console.print(f" → Saved to [bold blue]{seg_dir.relative_to(output_dir.parent)}/{folder_name}[/bold blue]")

    # ── NEW: Create combined_sound.wav with all speech segments concatenated ──
    if segments:
        console.print("\n[bold magenta]Creating combined_sound.wav from all speech segments...[/bold magenta]")

        combined_segments = []
        for seg in segments:
            start_sample = int(seg['start'] * 16000)
            end_sample = int(seg['end'] * 16000)
            segment_audio = waveform[:, start_sample:end_sample]
            combined_segments.append(segment_audio)
        
        # Concatenate along time dimension (dim=1)
        combined_audio = cat(combined_segments, dim=1)  # Shape: (1, total_samples)

        combined_path = output_dir / "combined_sound.wav"
        torchaudio.save(
            str(combined_path),
            combined_audio,
            sample_rate=16000,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        duration_sec = combined_audio.shape[1] / 16000
        console.print(f"Combined speech saved → [bold green]{combined_path.name}[/bold green] "
                      f"({duration_sec:.2f}s total speech)")

    # Save per-file timestamps (always, even if no segments — but we already returned early if none)
    save_file(segments, output_dir / "speech_timestamps.json")

# ── Updated __main__ block with summary collection ───────────────────────
if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    # audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/stream_speakers_extractor/speakers"
    # audio_paths = resolve_audio_paths(audio_dir, recursive=True)
    audio_paths = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251212_123859.wav"
    ]

    summary: Dict[str, Any] = {
        "total_files_processed": len(audio_paths),
        "files_with_speech": 0,
        "total_segments": 0,
        "per_file": defaultdict(dict),
    }

    # Ensure base OUTPUT_DIR exists for summary.json
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for audio_path in audio_paths:
        output_dir = OUTPUT_DIR

        # Run main – it will early-return if no segments
        main(audio_path, output_dir)

        # Count only files that actually produced segments
        if (output_dir / "speech_timestamps.json").exists():
            with open(output_dir / "speech_timestamps.json") as f:
                segs = json.load(f)
            count = len(segs)
            summary["files_with_speech"] += 1
            summary["total_segments"] += count
            summary["per_file"][str(audio_path)] = {"segments": count}

    # ── Write global summary.json in base OUTPUT_DIR ─────────────────────
    save_file(summary, Path(OUTPUT_DIR) / "summary.json")
    console.print(f"\n[bold green]Global summary saved to:[/bold green] {Path(OUTPUT_DIR)/'summary.json'}")