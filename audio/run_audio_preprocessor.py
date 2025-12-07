# run_audio_preprocessor.py
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

from jet.audio.speech.audio_preprocessor import AudioPreprocessor, PreprocessResult
import soundfile as sf
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# === OUTPUT DIRECTORY SETUP ===  ←←← YOU SAID: DO NOT CHANGE THIS
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_AUDIO_DIR = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic")


def save_result(audio_path: Path, result: PreprocessResult, subdir: Path):
    """Save preprocessed audio and JSON summary"""
    subdir.mkdir(parents=True, exist_ok=True)

    # Save preprocessed audio
    wav_path = subdir / "preprocessed.wav"
    sf.write(wav_path, result["audio"], result["sample_rate"])

    # Save summary
    summary = {
        "input_file": audio_path.name,
        "input_path": str(audio_path),
        "processed_at": datetime.now().isoformat(),
        "original_duration_sec": round(result["original_duration_sec"], 3),
        "output_duration_sec": round(result["duration_sec"], 3),
        "sample_rate_hz": result["sample_rate"],
        "vad_kept_ratio": round(result["vad_kept_ratio"], 4),
        "vad_kept_percent": f"{result['vad_kept_ratio']:.1%}",
        "silence_removed_sec": round(result["original_duration_sec"] - result["duration_sec"], 3),
    }

    with open(subdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    preprocessor = AudioPreprocessor(
        threshold=0.5,
        min_speech_duration=0.06,
        padding_duration=0.1,
    )

    rprint(Panel.fit(f"[bold cyan]Jet Audio Preprocessor[/bold cyan]\nOutput → [blue]{OUTPUT_DIR}[/]"))

    audio_dir = DEFAULT_AUDIO_DIR
    if not audio_dir.exists():
        rprint(f"[bold red]Input directory not found:[/] {audio_dir}")
        return

    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        rprint("[yellow]No .wav files found[/]")
        return

    table = Table(title="Processing Summary", show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Orig → Out", justify="right")
    table.add_column("Kept", justify="right")
    table.add_column("Removed", justify="right")
    table.add_column("Folder", style="dim")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Processing files...", total=len(wav_files))

        for audio_path in wav_files:
            progress.update(task, description=f"Processing [bold]{audio_path.name}[/]")

            # Create unique subfolder per file (uses stem + timestamp for uniqueness)
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in audio_path.stem)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            subdir_name = f"{safe_name}_{timestamp}"
            subdir = Path(OUTPUT_DIR) / subdir_name

            try:
                result = preprocessor.preprocess(str(audio_path), apply_vad=False)
                summary = save_result(audio_path, result, subdir)

                table.add_row(
                    audio_path.name,
                    f"{summary['original_duration_sec']}s → {summary['output_duration_sec']}s",
                    summary["vad_kept_percent"],
                    f"{summary['silence_removed_sec']}s",
                    subdir_name,
                )
            except Exception as e:
                rprint(f"[red]Failed:[/] {audio_path.name} → {e}")
                table.add_row(audio_path.name, "[red]ERROR", "-", "-", subdir_name)

            progress.advance(task)

    rprint(table)
    rprint(Panel.fit(f"[bold green]All done![/]\nResults saved to:\n[blue]{OUTPUT_DIR}[/]"))


if __name__ == "__main__":
    main()