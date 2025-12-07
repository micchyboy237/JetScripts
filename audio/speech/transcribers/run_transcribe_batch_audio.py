#!/usr/bin/env python
"""
Batch transcribe multiple audio files — works with a folder OR a single file.
Supports optional chunking for very long recordings.
Fast batch mode when possible, safe splitting when needed.
"""

import shutil
import argparse
from pathlib import Path

from faster_whisper import WhisperModel

from jet.audio.transcribers.utils import transcribe_batch_audio
from jet.audio.utils import split_audio


# ── Output directory: per-script generated folder, cleaned on each run ──
SCRIPT_DIR = Path(__file__).parent
OUTPUT_ROOT = SCRIPT_DIR / "generated" / Path(__file__).stem
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def transcribe_single_with_split(
    audio_path: Path,
    model: WhisperModel,
    segment_duration: float = 10.0,
    overlap: float = 1.0,
) -> str:
    """Transcribe one long file by splitting into overlapping chunks."""
    chunks = []
    for segment_audio, start, end in split_audio(
        str(audio_path),
        segment_duration=segment_duration,
        overlap_duration=overlap,
        sample_rate=16000,
    ):
        segments, _ = model.transcribe(segment_audio, word_timestamps=True, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        chunks.append(f"[{start:.1f}-{end:.1f}s] {text}")

    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files — accepts a folder or a single file"
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to a folder with audio files OR a single audio file (.wav, .mp3, .m4a, .flac, .ogg)",
    )
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=OUTPUT_ROOT / "transcripts",
        help="Output directory for transcript .txt files",
    )
    parser.add_argument(
        "--split-long",
        action="store_true",
        help="Split very long files into chunks (safer for >30–60 min recordings)",
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for GPU memory tuning")
    parser.add_argument("--segment-duration", type=float, default=10.0, help="Max seconds per chunk when splitting")
    parser.add_argument("--overlap", type=float, default=1.0, help="Overlap between chunks in seconds")
    args = parser.parse_args()

    # ── Resolve input: single file or directory → list of audio files ──
    if args.audio_path.is_file():
        audio_files = [args.audio_path]
        print(f"Single file mode: {args.audio_path.name}")
    elif args.audio_path.is_dir():
        audio_files = sorted(
            p
            for p in args.audio_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        )
        if not audio_files:
            print("No supported audio files found in the directory.")
            return
        print(f"Found {len(audio_files)} audio_files == 1 and 'file' or 'files' in folder")
    else:
        raise FileNotFoundError(f"Path not found or not a file/directory: {args.audio_path}")

    # ── Prepare output directory ──
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"Loading model '{args.model}' on {args.device}...")
    model = WhisperModel(args.model, device=args.device, compute_type="float16" if args.device == "cuda" else "int8")

    # ── Transcription mode ──
    if not args.split_long and len(audio_files) > 1:
        # Fast batch mode when possible
        print(f"Batch transcribing {len(audio_files)} files...")
        results = transcribe_batch_audio(
            [str(p) for p in audio_files],
            model,
            batch_size=args.batch_size,
            show_progress=True,
            progress_desc="Transcribing",
        )

        for audio_path, (segments, _) in zip(audio_files, results):
            text = "".join(seg.text for seg in segments).strip()
            out_path = args.output_dir / f"{audio_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"Saved → {out_path.name}")

    else:
        # Safe mode: one-by-one, with optional splitting
        mode = "with splitting" if args.split_long else "one-by-one"
        print(f"Transcribing {len(audio_files)} file(s) {mode}...")
        for audio_path in audio_files:
            print(f"Processing → {audio_path.name}")
            if args.split_long:
                text = transcribe_single_with_split(
                    audio_path, model, args.segment_duration, args.overlap
                )
            else:
                segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
                text = "".join(seg.text for seg in segments).strip()

            out_path = args.output_dir / f"{audio_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"Saved → {out_path.name}")

    print(f"\nAll done! Transcripts saved to:\n   {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()