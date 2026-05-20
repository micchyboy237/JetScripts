import argparse
import json
import shutil
from pathlib import Path

from jet.audio.helpers.config import FRAME_SHIFT_MS
from jet.audio.speech.vad_extractors import get_last_valley_trough, load_probs
from rich.console import Console
from rich.panel import Panel

console = Console()


def is_probs_list(obj):
    """Returns True if obj is a list of floats."""
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(x, float) for x in obj)
    )


def linkify(path: Path):
    return f"[link=file://{path}]{path.name}[/link]"


if __name__ == "__main__":
    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"

    parser = argparse.ArgumentParser(
        description=(
            "Find the valley trough that covers the last audio frame. "
            "Accepts an audio file, .npy VAD probs, or a JSON list of floats."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        type=str,
        help=(
            "Path to audio file (.wav, .mp3, etc.), "
            ".npy file of VAD probs, or JSON file/list of floats"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=Path(__file__).parent / "generated" / Path(__file__).stem,
        help="Output directory to save JSON results",
    )
    args = parser.parse_args()
    input_value = args.input

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        probs = None
        audio_np = None
        probs, audio_np = load_probs(input_value, DEFAULT_AUDIO)

        # ── Core call ────────────────────────────────────────────────────────
        frame_duration = FRAME_SHIFT_MS / 1000.0
        total_frames = len(probs) if probs else 0
        total_duration_s = total_frames * frame_duration

        last_trough = get_last_valley_trough(
            probs_or_audio=probs,
            smoothing_window=0,
            trough_height=0.3,
            trough_prominence=0.0,
            min_valley_duration_s=0.1,
            min_trough_offset_s=2.0,
        )

        output_file = args.output_dir / "last_valley_trough.json"

        # ── Display results ──────────────────────────────────────────────────
        if not last_trough:
            console.print(
                "[yellow]No valley trough found that covers the last audio frame.[/yellow]"
            )
        else:
            v = last_trough["valley"]
            percentage_offset = (
                round(last_trough["time_s"] / total_duration_s * 100, 2)
                if total_duration_s > 0
                else 0.0
            )

            console.print(
                Panel(
                    "[bold magenta]=== LAST VALLEY TROUGH ===[/bold magenta]",
                    expand=False,
                )
            )
            console.print(
                f"Time       : {last_trough['time_s']:.2f}s"
                f"  (Global: {last_trough.get('global_time_s', last_trough['time_s']):.2f}s)"
            )
            console.print(f"Percentage : [cyan]{percentage_offset}%[/cyan]")
            console.print(
                f"Prob       : {last_trough['prob']:.4f}"
                f" | Valley: {v['valley_score']:.4f}"
                f" | Trough: {v['trough_score']:.4f}"
            )
            console.print(
                f"Duration   : {v['duration_s']:.2f}s"
                f"  (frames {v['frame_start']}–{v['frame_end']})"
            )

            # is_last should always be True here, but we show it for clarity
            console.print(f"Is last    : [bold green]{v['is_last']}[/bold green]")

            output_data = dict(last_trough)
            output_data["percentage_offset"] = percentage_offset
            output_data["total_frames"] = total_frames
            output_data["total_duration_s"] = total_duration_s

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Results saved to:[/green] {linkify(output_file)}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
