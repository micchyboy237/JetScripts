import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from jet.audio.helpers.config import FRAME_SHIFT_MS
from jet.audio.speech.firered.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.vad_extractors import get_best_valley_trough
from jet.audio.utils.loader import load_audio
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
        description="Extract valley troughs (strong silence points) from audio file, .npy VAD probs, or JSON list of floats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Merge input to a single arg
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        type=str,
        help="Path to audio file (.wav, .mp3, etc.), .npy file of VAD probs, or JSON file/list of floats",
    )

    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=Path(__file__).parent / "generated" / Path(__file__).stem,
        help="Output directory to save JSON results (default: generated/)",
    )

    args = parser.parse_args()
    input_value = args.input

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Determine input type and load probs
        probs = None
        input_path = Path(input_value)
        if is_probs_list(input_value):
            # Passed as inline list of floats (not typical, but technically allowed)
            probs = input_value
            audio_np = None
        elif input_path.is_file():
            ext = input_path.suffix.lower()
            if ext == ".npy":
                console.print(f"Loading probabilities from: {linkify(input_path)}")
                np_load = np.load(input_path, allow_pickle=True)
                # Auto-handle npy with array or list
                if isinstance(np_load, np.ndarray):
                    np_load = np_load.tolist()
                probs = np_load
                audio_np = None
            elif ext in {".json", ".txt"}:
                console.print(f"Loading probabilities from: {linkify(input_path)}")
                with open(input_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if is_probs_list(loaded):
                    probs = loaded
                    audio_np = None
                else:
                    raise ValueError(
                        f"JSON file is not a list of floats: {linkify(input_path)}"
                    )
            else:
                # Treat any other file as audio
                console.print(f"Loading audio from: {linkify(input_path)}")
                audio = load_audio(input_path)
                if isinstance(audio, tuple) and len(audio) == 2:
                    audio_np = audio[0]
                else:
                    audio_np = audio
                _, probs = extract_speech_timestamps(
                    audio=audio_np,
                    threshold=0.5,
                    min_speech_duration_sec=0.250,
                    min_silence_duration_sec=0.250,
                    with_scores=True,
                )
        else:
            # Fallback: input provided as inline string/list? (could be args.input = '[0.2, 0.3, ...]')
            try:
                loaded = json.loads(input_value)
                if is_probs_list(loaded):
                    probs = loaded
                    audio_np = None
                else:
                    raise ValueError()
            except Exception:
                # If nothing else, treat as default audio path
                console.print(
                    f"[yellow]Input not recognized, falling back to default audio: {linkify(Path(DEFAULT_AUDIO))}[/yellow]"
                )
                audio = load_audio(DEFAULT_AUDIO)
                if isinstance(audio, tuple) and len(audio) == 2:
                    audio_np = audio[0]
                else:
                    audio_np = audio
                _, probs = extract_speech_timestamps(
                    audio=audio_np,
                    threshold=0.5,
                    min_speech_duration_sec=0.250,
                    min_silence_duration_sec=0.250,
                    with_scores=True,
                )

        frame_duration = FRAME_SHIFT_MS / 1000.0
        total_duration_s = len(probs) * frame_duration if probs else 0.0

        best_trough = get_best_valley_trough(
            probs=probs,
            min_valley_duration_s=0.8,
            smoothing_window=20,
            trough_prominence=0.15,
            valley_threshold=None,
            min_trough_offset_s=1.0,
        )

        output_file = args.output_dir / "best_valley_trough.json"

        if not best_trough:
            console.print("[yellow]No valid best valley trough found.[/yellow]")
        else:
            # Use a local var to store percentage (do not mutate ValleyTrough object directly!)
            percentage_offset = (
                round((best_trough["time_s"] / total_duration_s * 100), 2)
                if total_duration_s > 0
                else 0.0
            )

            console.print(
                Panel(
                    "[bold magenta]=== BEST VALLEY TROUGH ===[/bold magenta]",
                    expand=False,
                )
            )

            v = best_trough["valley"]
            console.print(
                f"Time       : {best_trough['time_s']:.2f}s  (Global: {best_trough.get('global_time_s', best_trough['time_s']):.2f}s)"
            )
            console.print(f"Percentage : [cyan]{percentage_offset}%[/cyan]")
            console.print(
                f"Prob       : {best_trough['prob']:.4f} | Valley: {v['valley_score']:.4f} | Trough: {v['trough_score']:.4f}"
            )
            console.print(f"Duration   : {v['duration_s']:.2f}s")

            # Save result including percentage_offset
            output_data = dict(best_trough)
            output_data["percentage_offset"] = percentage_offset
            if args.output_dir:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                console.print(
                    f"[green]Results saved to:[/green] {linkify(output_file)}"
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
