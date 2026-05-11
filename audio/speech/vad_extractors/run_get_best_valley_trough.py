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

if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"

    parser = argparse.ArgumentParser(
        description="Extract valley troughs (strong silence points) from audio or VAD probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options (arguments are now not required to allow default fallback)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--probs",
        "-p",
        type=Path,
        help="Path to .npy file containing VAD probabilities",
    )
    input_group.add_argument(
        "--audio",
        "-a",
        type=Path,
        default=DEFAULT_AUDIO,
        help="Path to audio file (.wav, .mp3, etc.)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=Path(__file__).parent / "generated" / Path(__file__).stem,
        help="Output directory to save JSON results (default: generated/)",
    )

    args = parser.parse_args()  # keep same CLI for easy use

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # If neither --probs nor --audio is provided, use DEFAULT_AUDIO
        if not getattr(args, "probs", None) and not getattr(args, "audio", None):
            console.print(
                f"No --audio or --probs provided, using default audio: {DEFAULT_AUDIO}"
            )
            args.audio = Path(DEFAULT_AUDIO)

        if args.probs:
            # Load probabilities from .npy file
            console.print(f"Loading probabilities from: {args.probs}")
            probs = np.load(args.probs)
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
        else:
            audio = load_audio(args.audio)
            # load_audio can return (audio, sr) tuple
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_np = audio[0]
            else:
                audio_np = audio
            _, probs = (
                extract_speech_timestamps(  # still need probs for the best-trough function
                    audio=audio_np,
                    threshold=0.5,
                    min_speech_duration_sec=0.250,
                    min_silence_duration_sec=0.250,
                    with_scores=True,
                )
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
            best_trough["percentage_offset"] = (
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
                f"Time       : {best_trough['time_s']:.3f}s  (Global: {best_trough.get('global_time_s', best_trough['time_s']):.3f}s)"
            )
            console.print(
                f"Percentage : [cyan]{best_trough['percentage_offset']}%[/cyan]"
            )
            console.print(
                f"Prob       : {best_trough['prob']:.4f} | Valley: {v['valley_score']:.4f} | Trough: {v['trough_score']:.4f}"
            )
            console.print(f"Duration   : {v['duration_s']:.3f}s")

            if args.output_dir:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(best_trough, f, indent=2, ensure_ascii=False)
                console.print(f"[green]Results saved to:[/green] {output_file}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
