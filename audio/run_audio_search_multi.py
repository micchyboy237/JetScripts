import sys

from jet.audio.audio_search import find_audio_offsets
from jet.audio.utils import load_audio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


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
    long_signal, sr_long = load_audio(long_path)
    short_signal, sr_short = load_audio(short_path)

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
