import sys

from jet.audio.audio_search import find_audio_offset
from jet.audio.utils import load_audio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ────────────────────────────────────────────────
#   Command-line usage example
# ────────────────────────────────────────────────

if len(sys.argv) != 3:
    console.print(
        Panel.fit(
            "[bold yellow]Usage:[/]\n"
            "  python this_file.py  LONG_AUDIO.wav  SHORT_CLIP.wav\n\n"
            "Example:\n"
            "  python this_file.py podcast_episode_45.wav   jingle.wav",
            title="Audio Offset Finder",
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
        # You could resample here if needed — omitted for simplicity

    console.rule("Searching for short clip inside long audio")
    result = find_audio_offset(
        long_signal=long_signal,
        short_signal=short_signal,
        sample_rate=sr_long,
        confidence_threshold=0.78,  # ← feel free to tune
    )

    if result is None:
        console.print("[bold red]No good match found[/] (confidence below threshold)")
    else:
        table = Table(show_header=False, expand=False)
        table.add_row("Sample rate", f"{sr_long:,d} Hz")
        table.add_row("Start sample", f"{result['start_sample']:,d}")
        table.add_row("End sample", f"{result['end_sample']:,d}")
        table.add_row("Start time", f"{result['start_time']:.3f} s")
        table.add_row("End time", f"{result['end_time']:.3f} s")
        table.add_row("Confidence", f"[bold green]{result['confidence']:.4f}[/]")

        console.print(
            Panel(
                table,
                title="Best match found (threshold ≥ 0.78)",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Optional: show rough location in timeline
        duration_long = len(long_signal) / sr_long
        console.print(f"Long audio duration: [cyan]{duration_long:.1f} seconds[/]")

except Exception as e:
    console.print(f"\n[bold red]Error:[/] {e}", style="red")
    sys.exit(1)
