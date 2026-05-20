import argparse
import shutil
from pathlib import Path

from jet.audio.audio_waveform.vad.vad_speech_splitter import split_segments
from jet.audio.audio_waveform.vad.vad_utils import save_segments
from jet.audio.speech.utils import display_segments
from jet.audio.utils.base import extract_audio_segment
from jet.audio.utils.loader import load_audio
from rich.console import Console

console = Console()


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

    parser = argparse.ArgumentParser(
        description="Extract valley troughs (strong silence points) from audio file, .npy VAD probs, or JSON list of floats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        type=str,
        help="Path to audio file (.wav, .mp3, etc.), .npy file of VAD probs, or JSON file/list of floats",
    )

    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory to save JSON results (default: generated/)",
    )

    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_seg_limit_s = 8.0

    try:
        audio_np, sr = load_audio(args.audio_path)
        left_audio_np, _ = extract_audio_segment(audio_np, end=max_seg_limit_s)

        segments, audio_chunks = split_segments(left_audio_np)

        saved_segments = save_segments(
            segments=segments,
            audio_chunks=audio_chunks,
            output_base_dir=args.output_dir,
            show_progress=True,
        )

        display_segments(segments)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
