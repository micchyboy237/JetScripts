#!/usr/bin/env python
"""
Single-file transcription with automatic chunking to avoid OOM and improve accuracy.
Perfect for long recordings (podcasts, meetings, interviews).
"""

import os
import shutil
import argparse
from pathlib import Path

from faster_whisper import WhisperModel
import re

from faster_whisper.transcribe import Segment
from jet.audio.transcribers.utils import transcribe_audio
from jet.audio.utils import split_audio
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Transcribe a single long audio file with chunking")
    parser.add_argument("audio_path", type=Path, help="Path to input audio file")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--segment-duration", type=float, default=20.0, help="Max seconds per chunk")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap between chunks (seconds)")
    args = parser.parse_args()

    print(f"Loading model: {args.model} on {args.device}")
    model = WhisperModel(args.model, device=args.device)

    print(f"Splitting and transcribing: {args.audio_path.name}")
    
    chunks = []
    all_global_segments = []  # To collect all segments with absolute timestamps

    for i, (segment_audio, start, end) in enumerate(
        split_audio(
            str(args.audio_path),
            segment_duration=args.segment_duration,
            overlap_duration=args.overlap,
            sample_rate=16000,
        ),
        start=1,
    ):
        print(f"  → Chunk {i}: {start:.1f}s → {end:.1f}s")
        segments, info = transcribe_audio(
            segment_audio, model, language="ja", task="translate", show_progress=True
        )

        # Convert chunk-relative segments to global time
        for seg in segments:
            # Create new Segment with absolute timestamps
            seg_dict = seg._asdict()
            seg_dict["start"] += start
            seg_dict["end"] += start
            global_seg = Segment(**seg_dict)
            all_global_segments.append(global_seg)

        # Still save per-chunk plain text for debugging
        chunk_text = " ".join(seg.text.strip() for seg in segments)
        chunk_file = args.output_dir / f"chunk_{i}.txt"
        chunk_file.write_text(chunk_text, encoding="utf-8")
        print(f"Chunk saved to {chunk_file}")

        chunks.append({
            "chunk_idx": i,
            "start": start,
            "end": end,
            "chunk_text": chunk_text,
        })
        save_file(segments, f"{OUTPUT_DIR}/chunk_{i}/segments.json")
        save_file(info, f"{OUTPUT_DIR}/chunk_{i}/info.json")
        save_file(chunk_text, f"{OUTPUT_DIR}/chunk_{i}/text.txt")
        save_file({
            "chunk_idx": i,
            "start": start,
            "end": end,
        }, f"{OUTPUT_DIR}/chunk_{i}/meta.json")
        save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    # === Consolidate overlaps and generate clean final transcript ===
    merged_segments = []

    if all_global_segments:
        # Sort by start time (defensive)
        all_global_segments.sort(key=lambda s: s.start)

        current = all_global_segments[0]

        for next_seg in all_global_segments[1:]:
            # If segments are close in time and text connects naturally
            time_gap = next_seg.start - current.end
            if time_gap < 0.6:  # within 600ms → likely same sentence
                # Merge text (add space if needed)
                merged_text = current.text
                if not re.search(r'[。！？.!?]\s*$', current.text.strip()):
                    merged_text += " " + next_seg.text.strip()
                else:
                    merged_text += next_seg.text.strip()

                # Create new Segment with updated text and end time
                current_dict = current._asdict()
                current_dict["text"] = merged_text
                current_dict["end"] = next_seg.end
                current = Segment(**current_dict)
                continue

            # No merge → save current and move to next
            merged_segments.append(current)
            current = next_seg

        merged_segments.append(current)  # last segment

        # Build final transcript
        lines = []
        for seg in merged_segments:
            timestamp = f"[{seg.start:.2f}s - {seg.end:.2f}s]"
            text = seg.text.strip()
            if text:
                lines.append(f"{timestamp} {text}")

        final_transcript = "\n\n".join(lines)
    else:
        final_transcript = ""

    print(f"Consolidated transcript: {len(all_global_segments)} → {len(merged_segments)} segments")
    
    transcription_file = args.output_dir / "transcription.txt"
    transcription_file.write_text(final_transcript, encoding="utf-8")
    print(f"Final cleaned transcription saved to {transcription_file}")


if __name__ == "__main__":
    main()