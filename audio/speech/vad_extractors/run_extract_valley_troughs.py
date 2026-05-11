import numpy as np
from jet.audio.helpers.config import FRAME_SHIFT_MS
from jet.audio.speech.firered.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.vad_extractors import extract_valley_troughs
from jet.audio.utils.loader import load_audio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

    # Input options
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
        help="Output directory to save JSON results",
    )

    args = parser.parse_args()

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
            console.print(f"Loading probabilities from: {args.probs}")
            probs = np.load(args.probs)
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
        else:
            audio = load_audio(args.audio)
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

        # Compute total duration
        frame_duration = FRAME_SHIFT_MS / 1000.0
        total_duration_s = len(probs) * frame_duration if probs else 0.0

        troughs = extract_valley_troughs(
            probs=probs,
            min_valley_duration_s=0.8,
            smoothing_window=20,
            trough_prominence=0.15,
            valley_threshold=None,
            min_trough_offset_s=1.0,
        )

        # Enrich with percentage offset
        for trough in troughs:
            t = trough["time_s"]
            trough["percentage_offset"] = (
                round((t / total_duration_s * 100), 2) if total_duration_s > 0 else 0.0
            )

        # === Add Rank based on final_score (higher = better) ===
        sorted_troughs = sorted(
            troughs, key=lambda x: x["valley"]["final_score"], reverse=True
        )
        rank_map = {
            trough["time_s"]: rank for rank, trough in enumerate(sorted_troughs, 1)
        }

        for trough in troughs:
            trough["rank"] = rank_map[trough["time_s"]]
        # =======================================================

        # Output results
        if not troughs:
            console.print("No valid valley troughs found.")
        else:
            # Enhanced Panel with Total Duration
            total_min = int(total_duration_s // 60)
            total_sec = total_duration_s % 60
            duration_str = (
                f"{total_min:02d}:{total_sec:05.2f}"
                if total_min > 0
                else f"{total_duration_s:.2f}s"
            )

            console.print(
                Panel(
                    f"[bold green]Found {len(troughs)} Valley Trough(s)[/bold green] "
                    f"• [bold white]Total Duration:[/bold white] [cyan]{duration_str}[/cyan]",
                    expand=False,
                )
            )

            table = Table(
                title="Valley Troughs", show_header=True, header_style="bold cyan"
            )
            table.add_column("Rank", style="bold", justify="center")
            table.add_column("Time (s)", justify="right")
            table.add_column("% Offset", justify="right")
            table.add_column("Prob", justify="right")
            table.add_column("Final Score", justify="right")
            table.add_column("Duration (s)", justify="right")

            for trough in troughs:
                v = trough["valley"]
                rank = trough["rank"]

                # Color coding for ranks
                if rank == 1:
                    row_style = "bold green"  # Rank 1
                elif rank == 2:
                    row_style = "bold yellow"  # Rank 2
                else:
                    row_style = "bold orange1"  # Rank 3+

                table.add_row(
                    str(rank),
                    f"{trough['time_s']:.3f}",
                    f"{trough['percentage_offset']}%",
                    f"{trough['prob']:.4f}",
                    f"{v['final_score']:.4f}",
                    f"{v['duration_s']:.3f}",
                    style=row_style,
                )

            console.print(table)

            # Save to JSON
            output_file = args.output_dir / "valley_troughs.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(troughs, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Results saved to:[/green] {output_file}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
