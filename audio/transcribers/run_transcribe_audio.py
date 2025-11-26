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

from jet.audio.transcribers.utils import transcribe_audio
from jet.audio.utils import split_audio

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Transcribe a single long audio file with chunking")
    parser.add_argument("audio_path", type=Path, help="Path to input audio file")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--output", "-o", type=Path, default=Path(f"{OUTPUT_DIR}/transcription.txt"))
    parser.add_argument("--segment-duration", type=float, default=10.0, help="Max seconds per chunk")
    parser.add_argument("--overlap", type=float, default=1.0, help="Overlap between chunks (seconds)")
    args = parser.parse_args()

    print(f"Loading model: {args.model} on {args.device}")
    model = WhisperModel(args.model, device=args.device)

    full_text = []
    print(f"Splitting and transcribing: {args.audio_path.name}")
    
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
        segments, _ = transcribe_audio(segment_audio, model, language="en", show_progress=True)
        
        chunk_text = " ".join(seg.text.strip() for seg in segments)
        full_text.append(f"[{start:.1f}s - {end:.1f}s] {chunk_text}")

    final_transcript = "\n\n".join(full_text)
    
    args.output.write_text(final_transcript, encoding="utf-8")
    print(f"Transcription saved to {args.output}")


if __name__ == "__main__":
    main()