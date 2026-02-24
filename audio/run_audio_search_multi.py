import sys
from pathlib import Path

import numpy as np
from jet.audio.audio_search import find_audio_offsets
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.io import wavfile

console = Console()


def load_wav_mono(path: str | Path) -> tuple[int, np.ndarray]:
    """Load a WAV file and ensure it's mono float32 normalized to [-1, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    sample_rate, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu":
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return sample_rate, data


if len(sys.argv) != 3:
    console.print(
        Panel.fit(
            "[bold yellow]Usage:[/]\n"
            " python run_audio_search_multi.py LONG_AUDIO.wav SHORT_CLIP.wav\n\n"
            "Finds all occurrences of the short clip inside the long audio.\n\n"
            "Example:\n"
            " python run_audio_search_multi.py podcast_episode_45.wav jingle.wav",
            title="Multi Audio Offset Finder",
            border_style="bright_blue",
        )
    )
    sys.exit(1)

long_path = sys.argv[1]
short_path = sys.argv[2]

try:
    console.rule("Loading audio files")
    sr_long, long_signal = load_wav_mono(long_path)
    sr_short, short_signal = load_wav_mono(short_path)

    if sr_long != sr_short:
        console.print(
            f"[red]Warning:[/] Sample rates do not match "
            f"({sr_long} Hz vs {sr_short} Hz). Using {sr_long} Hz."
        )

    console.rule("Searching for all occurrences of short clip")

    matches = find_audio_offsets(
        long_signal=long_signal,
        short_signal=short_signal,
        sample_rate=sr_long,
        confidence_threshold=0.78,
        verbose=True,
    )

    duration_long = len(long_signal) / sr_long
    console.print(f"Long audio duration: [cyan]{duration_long:.1f} seconds[/]")

    if not matches:
        console.print("[bold red]No good matches found[/] (confidence < 0.78)")
        console.print("[dim]Try lowering --threshold if you suspect weak matches[/]")
    else:
        title = (
            "Multiple matches found (threshold ≥ 0.78)"
            if len(matches) > 1
            else "Match found (threshold ≥ 0.78)"
        )
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Start time", justify="right")
        table.add_column("End time", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Start sample", justify="right")
        table.add_column("Confidence", justify="right", style="green")

        for i, m in enumerate(matches, 1):
            table.add_row(
                str(i),
                f"{m['start_time']:.3f} s",
                f"{m['end_time']:.3f} s",
                f"{m['end_time'] - m['start_time']:.3f} s",
                f"{m['start_sample']:,d}",
                f"{m['confidence']:.4f}",
            )

        console.print(Panel(table, title=title, border_style="green", padding=(1, 2)))

except Exception as e:
    console.print(f"\n[bold red]Error:[/] {e}", style="red")
    sys.exit(1)
