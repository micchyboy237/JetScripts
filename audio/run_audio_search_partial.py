import argparse
import sys
from pathlib import Path

from jet.audio.audio_search import (
    find_partial_audio_matches,
)
from jet.audio.utils import load_audio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Find partial / substring occurrences of a short audio clip inside a longer audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("long_audio", type=str, help="Path to long audio file (WAV)")
    parser.add_argument(
        "short_clip", type=str, help="Path to short clip to search for (WAV)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum confidence threshold for partial matches (default: 0.75)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.50,
        metavar="FRAC",
        help="Minimum fraction of short clip that must match (default: 0.50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster mode: coarser steps, fewer sub-clips",
    )

    args = parser.parse_args()

    long_path = Path(args.long_audio)
    short_path = Path(args.short_clip)

    try:
        console.rule("Loading audio files")
        long_signal, sr_long = load_audio(long_path)
        short_signal, sr_short = load_audio(short_path)

        if sr_long != sr_short:
            console.print(
                f"[red]Warning:[/] Sample rates differ "
                f"({sr_long} Hz vs {sr_short} Hz). Using {sr_long} Hz for timing."
            )

        duration_long = len(long_signal) / sr_long
        duration_short = len(short_signal) / sr_short

        console.print(f"Long audio:  {duration_long:.1f} seconds")
        console.print(f"Short clip:  {duration_short:.1f} seconds")

        console.rule("Searching for partial matches")

        matches = find_partial_audio_matches(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=sr_long,
            verbose=True,
            confidence_threshold=args.threshold,
            min_match_fraction=args.min_fraction,
            length_step_fraction=0.18 if args.quick else 0.10,
            max_subclips=35 if args.quick else 80,
            # You can expose more params later if needed (step, max_fraction, etc.)
        )

        if not matches:
            console.print(
                f"[bold red]No partial matches found[/] above confidence {args.threshold:.2f} "
                f"and min length fraction {args.min_fraction:.2f}"
            )
            console.print("[dim]Try lowering --threshold or --min-fraction[/]")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Start", justify="right")
        table.add_column("End", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Matched frac", justify="right")
        table.add_column("Confidence", justify="right", style="green")

        for i, m in enumerate(matches, 1):
            start_t = m.start_sample / sr_long
            end_t = m.end_sample / sr_long
            dur = end_t - start_t
            matched_frac = m.match_length_samples / len(short_signal)

            table.add_row(
                str(i),
                f"{start_t:.3f} s",
                f"{end_t:.3f} s",
                f"{dur:.3f} s",
                f"{matched_frac:.2%}",
                f"{m.confidence:.4f}",
            )

        title = "Partial matches found" if len(matches) > 1 else "Partial match found"
        console.print(
            Panel(
                table,
                title=f"{title} (threshold ≥ {args.threshold:.2f})",
                border_style="green",
                padding=(1, 2),
            )
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
