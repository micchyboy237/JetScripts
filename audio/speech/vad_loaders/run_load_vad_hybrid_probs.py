import argparse
import json
import shutil
from pathlib import Path

from jet.audio.audio_waveform.vad.vad_utils import save_segments
from jet.audio.speech.utils import display_segments
from jet.audio.speech.vad_loaders import get_global_vad, load_vad_hybrid_probs
from jet.audio.utils.base import extract_audio_segment
from jet.audio.utils.loader import load_audio
from jet.transformers.object import make_serializable
from rich.console import Console

console = Console()
SAMPLE_RATE = 16_000


def linkify(path: Path):
    return f"[link=file://{path}]{path.name}[/link]"


if __name__ == "__main__":
    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"
    parser = argparse.ArgumentParser(
        description="Split VAD probabilities into two halves at the best valley trough.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        type=str,
        help="Path to audio file (.wav, .mp3, etc.)",
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

    max_seg_limit_s = None
    audio_np, sr = load_audio(args.audio_path)
    left_audio_np, _ = extract_audio_segment(audio_np, end=max_seg_limit_s)

    hybrid_probs, data = load_vad_hybrid_probs(left_audio_np)

    output_file = args.output_dir / "hybrid_probs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hybrid_probs, f, indent=2, ensure_ascii=False)
    console.print(f"[green]hybrid_probs saved to:[/green] {linkify(output_file)}")

    output_file = args.output_dir / "result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(data["result"]), f, indent=2, ensure_ascii=False)
    console.print(f"[green]result saved to:[/green] {linkify(output_file)}")

    output_file = args.output_dir / "frame_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            make_serializable(data["frame_results"]), f, indent=2, ensure_ascii=False
        )
    console.print(f"[green]frame_results saved to:[/green] {linkify(output_file)}")

    vad = get_global_vad()

    # Unpack tuple: segments + matched audio chunks in one call
    speech_segments, audio_chunks = vad.get_speech_segments(
        return_seconds=False,
        audio_np=left_audio_np,
    )

    output_file = args.output_dir / "speech_segments.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(speech_segments), f, indent=2, ensure_ascii=False)
    console.print(f"[green]speech_segments saved to:[/green] {linkify(output_file)}")

    save_segments(
        segments=speech_segments,
        audio_chunks=audio_chunks,
        output_base_dir=args.output_dir,
        show_progress=True,
        is_already_hybrid=True,
    )
    display_segments(speech_segments, time_format="ms")
