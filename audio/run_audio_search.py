import sys
from pathlib import Path

import numpy as np
from jet.audio.audio_search import find_audio_offset
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
        # Convert stereo → mono by averaging channels
        data = data.mean(axis=1)

    # Convert to float32 [-1, 1]
    if data.dtype.kind in "iu":  # integer types
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return sample_rate, data


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
    sr_long, long_signal = load_wav_mono(long_path)
    sr_short, short_signal = load_wav_mono(short_path)

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
