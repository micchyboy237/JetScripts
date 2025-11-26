#!/usr/bin/env python
"""
Perfect long-audio transcription: chunked, overlap-aware, zero duplicates,
and beautifully formatted timestamps (always 2 decimals).
"""

import shutil
import argparse
import json
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from jet.audio.transcribers.utils import transcribe_audio
from jet.audio.utils import split_audio
from jet.file.utils import save_file


# Output folder – auto-cleaned each run
OUTPUT_DIR = Path(__file__).with_name("generated") / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """Always return time with exactly 2 decimal places (e.g. 6.80, 18.60)"""
    return f"{seconds:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe long audio – clean, fast, no duplicates")
    parser.add_argument("audio_path", type=Path, help="Input audio file")
    parser.add_argument("--model", default="large-v3", help="Whisper model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", "-o", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--segment-duration", type=float, default=20.0, help="Seconds per chunk")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap in seconds")
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    model = WhisperModel(args.model, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Transcribing: {args.audio_path.name}")
    print(f"Chunk: {args.segment_duration}s | Overlap: {args.overlap}s\n")

    final_segments: List[Segment] = []
    keep_from_time: float = 0.0

    for i, item in enumerate(
        split_audio(
            str(args.audio_path),
            segment_duration=args.segment_duration,
            overlap_duration=args.overlap,
            sample_rate=16000,
        ),
        start=1,
    ):
        start_sec = item["start_time"]
        end_sec = item["end_time"]
        overlaps_prev = item["overlaps_previous"]

        print(f"Chunk {i:2d} | {format_time(start_sec)}s → {format_time(end_sec)}s | prev_overlap={overlaps_prev}")

        segments, info = transcribe_audio(
            item["segment"],
            model,
            language="ja",
            task="translate",
            show_progress=False,
        )

        # Save per-chunk debug data
        chunk_dir = args.output_dir / f"chunk_{i}"
        chunk_dir.mkdir(exist_ok=True)
        save_file(segments, chunk_dir / "segments.json")
        save_file(info, chunk_dir / "info.json")
        save_file(" ".join(s.text.strip() for s in segments), chunk_dir / "text.txt")
        save_file({**item, "chunk_idx": i}, chunk_dir / "meta.json")

        # Precise overlap trimming
        trim_until = start_sec + args.overlap if overlaps_prev else None

        for seg in segments:
            seg_global_start = start_sec + seg.start
            seg_global_end   = start_sec + seg.end

            if trim_until and seg_global_end <= trim_until:
                continue
            if trim_until and seg_global_start < trim_until:
                seg_global_start = trim_until

            if seg_global_start >= keep_from_time:
                new_seg = seg._asdict()
                new_seg["start"] = seg_global_start
                new_seg["end"]   = seg_global_end
                final_segments.append(Segment(**new_seg))
                keep_from_time = seg_global_end

    # Final clean transcript with perfect 2-decimal timestamps
    if final_segments:
        final_segments.sort(key=lambda s: s.start)
        lines = [
            f"[{format_time(s.start)}s - {format_time(s.end)}s] {s.text.strip()}"
            for s in final_segments
            if s.text.strip()
        ]
        final_transcript = "\n\n".join(lines)
    else:
        final_transcript = ""

    transcription_file = args.output_dir / "transcription.txt"
    transcription_file.write_text(final_transcript, encoding="utf-8")

    # Accurate summary
    raw_count = sum(
        len(json.loads(p.read_text(encoding="utf-8")))
        for p in args.output_dir.glob("chunk_*/segments.json")
        if p.is_file()
    )

    print("\n" + "="*70)
    print("TRANSCRIPTION COMPLETE")
    print(f"Raw segments (before deduplication)  : {raw_count}")
    print(f"Final segments (clean, no duplicates): {len(final_segments)}")
    print(f"Output → {transcription_file.resolve()}")
    print("="*70)


if __name__ == "__main__":
    main()