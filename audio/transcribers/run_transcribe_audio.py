import os
import shutil
from typing import List
from faster_whisper.transcribe import Segment
from jet.audio.transcribers.utils import transcribe_audio

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_022942.wav"

    segments_gen, info = transcribe_audio(
        audio_path=audio_path,
        model_name="large-v3",
        device="auto",                    # ← Will use "mps" on your Mac M1
        compute_type="int8_float32",           # ← Fast & accurate
        overlap_seconds=2.0,              # ← Critical fix: was 2.0 → now 8s
        chunk_length_seconds=20,
        vad_filter=True,
        word_timestamps=True,
        language="ja",
        task="translate",                 # Japanese → English
        beam_size=5,
        temperature=[0.0, 0.2, 0.4],
        repetition_penalty=1.1,
        condition_on_previous_text=True,
    )

    print("\n" + "="*70)
    print(f"Detected Language : {info.language.upper()} ({info.language_probability:.1%})")
    print(f"Audio Duration    : {info.duration / 60:.2f} minutes")
    print("Model Used        : large-v3")
    print("Task              : translate (Japanese → English)")
    print("="*70 + "\n")

    # FIXED: Don't call generator with () — just iterate
    all_segments = list(segments_gen)  # Materialize once for printing + SRT

    for seg in all_segments:
        print(f"[{seg.start:>7.2f} → {seg.end:>7.2f}] {seg.text}")

    # Save SRT (safe: reuses already-materialized segments)
    def save_as_srt(segments: List[Segment], output_path: str):
        os.makedirs(os.path.dirname(srt_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = f"{int(seg.start//3600):02d}:{int(seg.start%3600//60):02d}:{seg.start%60:06.3f}"[:-3]
                end = f"{int(seg.end//3600):02d}:{int(seg.end%3600//60):02d}:{seg.end%60:06.3f}"[:-3]
                f.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")

    srt_filename = os.path.basename(audio_path).replace(".wav", "_translated.srt")
    srt_path = f"{OUTPUT_DIR}/{srt_filename}"
    save_as_srt(all_segments, srt_path)
    print(f"\nSRT saved → {srt_path}")