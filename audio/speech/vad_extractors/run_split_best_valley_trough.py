import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from jet.audio.helpers.config import FRAME_SHIFT_MS
from jet.audio.speech.firered.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.vad_extractors import split_best_valley_trough
from jet.audio.utils.loader import load_audio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
SAMPLE_RATE = 16_000


def linkify(path: Path):
    # Provide clickable file link with basename (for rich/terminal that support it)
    return f"[link=file://{path}]{path.name}[/link]"


if __name__ == "__main__":
    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"

    parser = argparse.ArgumentParser(
        description="Split VAD probabilities into two halves at the best valley trough.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="Output directory to save JSON results",
    )
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def is_probs_list(obj):
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and all(isinstance(x, float) for x in obj)
        )

    try:
        input_value = args.input
        input_path = Path(input_value)
        audio_np = None
        sr = SAMPLE_RATE

        # Try to load from a recognized input type
        probs = None
        if is_probs_list(input_value):
            probs = input_value
        elif input_path.is_file():
            ext = input_path.suffix.lower()
            if ext == ".npy":
                console.print(f"Loading probabilities from: {input_path}")
                np_load = np.load(input_path, allow_pickle=True)
                probs = np_load.tolist() if isinstance(np_load, np.ndarray) else np_load
            elif ext in {".json", ".txt"}:
                console.print(f"Loading probabilities from: {input_path}")
                with open(input_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                probs = loaded if is_probs_list(loaded) else None
            else:
                console.print(f"Loading audio from: {input_path}")
                audio_np, sr = load_audio(input_path, SAMPLE_RATE)
                _, probs = extract_speech_timestamps(
                    audio=audio_np,
                    threshold=0.5,
                    min_speech_duration_sec=0.250,
                    min_silence_duration_sec=0.250,
                    with_scores=True,
                )
        else:
            try:
                loaded = json.loads(input_value)
                probs = loaded if is_probs_list(loaded) else None
            except Exception:
                console.print(
                    f"[yellow]Input not recognized, falling back to default audio: {DEFAULT_AUDIO}[/yellow]"
                )
                audio_np, sr = load_audio(DEFAULT_AUDIO, SAMPLE_RATE)
                _, probs = extract_speech_timestamps(
                    audio=audio_np,
                    threshold=0.5,
                    min_speech_duration_sec=0.250,
                    min_silence_duration_sec=0.250,
                    with_scores=True,
                )

        frame_duration = FRAME_SHIFT_MS / 1000.0
        total_duration_s = len(probs) * frame_duration if probs else 0.0

        result = split_best_valley_trough(
            probs=probs,
            min_valley_duration_s=0.8,
            smoothing_window=20,
            trough_prominence=0.15,
            valley_threshold=None,
            min_trough_offset_s=1.0,
        )

        if result is None:
            console.print(
                "[yellow]No suitable valley trough found — cannot split.[/yellow]"
            )
        else:
            (left_probs, best_trough), (right_probs, _) = result

            split_frame = best_trough["frame"]
            split_time_s = best_trough["time_s"]
            global_time_s = best_trough.get("global_time_s", split_time_s)
            split_pct = (
                round(split_time_s / total_duration_s * 100, 2)
                if total_duration_s > 0
                else 0.0
            )

            left_duration_s = len(left_probs) * frame_duration
            right_duration_s = len(right_probs) * frame_duration

            v = best_trough["valley"]

            # ── Header panel ────────────────────────────────────────────────
            console.print(
                Panel(
                    f"[bold green]Split at frame {split_frame}[/bold green]"
                    f"  •  [cyan]{split_time_s:.3f}s[/cyan]"
                    f"  •  [yellow]{split_pct}%[/yellow] of total"
                    f"  •  Score: [magenta]{v['final_score']:.4f}[/magenta]",
                    title="[bold]Best Valley Trough Split[/bold]",
                    expand=False,
                )
            )

            # ── Trough detail block (mirrors run_get_best_valley_trough) ────
            console.print(
                f"Time       : {split_time_s:.3f}s  (Global: {global_time_s:.3f}s)"
            )
            console.print(f"Percentage : [cyan]{split_pct}%[/cyan]")
            console.print(
                f"Prob       : {best_trough['prob']:.4f}"
                f" | Valley: {v['valley_score']:.4f}"
                f" | Trough: {v['trough_score']:.4f}"
            )
            console.print(f"Duration   : {v['duration_s']:.3f}s")

            # ── Left / right summary table ───────────────────────────────────
            table = Table(
                title="Split Halves",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Half", style="bold", justify="left")
            table.add_column("Frames", justify="right")
            table.add_column("Duration (s)", justify="right")
            table.add_column("Start frame", justify="right")
            table.add_column("End frame", justify="right")

            table.add_row(
                "Left  (before split)",
                str(len(left_probs)),
                f"{left_duration_s:.3f}",
                "0",
                str(split_frame - 1),
                style="bold green",
            )
            table.add_row(
                "Right (from split)",
                str(len(right_probs)),
                f"{right_duration_s:.3f}",
                str(split_frame),
                str(len(probs) - 1),
                style="bold yellow",
            )
            console.print(table)

            # ── Save splitted wav files (only when audio was loaded) ─────────

            left_wav_path = None
            right_wav_path = None

            if audio_np is not None:
                seconds_per_frame = FRAME_SHIFT_MS / 1000.0
                split_sample = int(split_frame * seconds_per_frame * sr)
                split_sample = max(0, min(split_sample, len(audio_np)))

                left_audio = audio_np[:split_sample]
                right_audio = audio_np[split_sample:]

                splitted_dir = args.output_dir / "splitted_wavs"
                splitted_dir.mkdir(parents=True, exist_ok=True)

                left_wav_path = splitted_dir / "left.wav"
                right_wav_path = splitted_dir / "right.wav"

                sf.write(str(left_wav_path), left_audio, sr, subtype="FLOAT")
                sf.write(str(right_wav_path), right_audio, sr, subtype="FLOAT")

                left_samples = len(left_audio)
                right_samples = len(right_audio)

                wav_table = Table(
                    title="Saved WAV Files",
                    show_header=True,
                    header_style="bold cyan",
                )
                wav_table.add_column("Half", style="bold", justify="left")
                wav_table.add_column("Samples", justify="right")
                wav_table.add_column("Duration (s)", justify="right")
                wav_table.add_column("Path", justify="left")

                wav_table.add_row(
                    "Left",
                    str(left_samples),
                    f"{left_samples / sr:.3f}",
                    linkify(left_wav_path),
                    style="bold green",
                )
                wav_table.add_row(
                    "Right",
                    str(right_samples),
                    f"{right_samples / sr:.3f}",
                    linkify(right_wav_path),
                    style="bold yellow",
                )

                console.print(wav_table)
            else:
                console.print(
                    "[yellow]Skipping wav export — no audio loaded "
                    "(use --audio instead of --probs to enable).[/yellow]"
                )

            # ── Save JSON ────────────────────────────────────────────────────
            output = {
                "split_frame": split_frame,
                "split_time_s": split_time_s,
                "global_time_s": global_time_s,
                "split_percentage": split_pct,
                "best_trough": best_trough,
                "left": {
                    "frame_start": 0,
                    "frame_end": split_frame - 1,
                    "num_frames": len(left_probs),
                    "duration_s": left_duration_s,
                },
                "right": {
                    "frame_start": split_frame,
                    "frame_end": len(probs) - 1,
                    "num_frames": len(right_probs),
                    "duration_s": right_duration_s,
                },
                "wavs": {
                    "left": str(left_wav_path) if left_wav_path else None,
                    "right": str(right_wav_path) if right_wav_path else None,
                },
            }

            output_file = args.output_dir / "split_best_valley_trough.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Results saved to:[/green] {output_file}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
