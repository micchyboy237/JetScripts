# audio_duration.py
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def get_audio_duration(path: str | Path) -> float:
    """
    Returns duration of audio file in seconds.

    Args:
        path: Path to audio file (mp3, wav, flac, m4a, ogg, opus, aac, ...)

    Returns:
        Duration in seconds (float)

    Raises:
        FileNotFoundError: if file doesn't exist
        CouldntDecodeError: if file format is not supported / corrupted
        ValueError: for other unexpected cases
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio = AudioSegment.from_file(str(path))
        # pydub uses milliseconds internally
        return audio.duration_seconds
    except CouldntDecodeError as e:
        raise CouldntDecodeError(f"Cannot decode audio file {path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to get duration of {path}: {e}") from e


def get_audio_duration_friendly(path: str | Path) -> str:
    """Returns human-readable duration string like '3:45' or '1:12:09'"""
    seconds = get_audio_duration(path)
    return seconds_to_hms(seconds)


def seconds_to_hms(total_seconds: float) -> str:
    """Convert float seconds → human readable mm:ss or h:mm:ss"""
    if total_seconds < 0:
        return "0:00"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int((total_seconds % 1) * 1000)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"


# ────────────────────────────────────────
#          Quick CLI usage example
# ────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    def main():
        import sys

        if len(sys.argv) < 2:
            console.print("[yellow]Usage:[/] python audio_duration.py <path_to_audio>")
            sys.exit(1)

        file_path = sys.argv[1]

        try:
            duration_sec = get_audio_duration(file_path)
            friendly = get_audio_duration_friendly(file_path)
            console.print(
                f"[cyan]{file_path}[/]: [green]{friendly}[/] ([dim]{duration_sec:.3f} s[/])"
            )
        except Exception as e:
            console.print(f"[red]Error:[/] {e}", style="bold")

    main()
