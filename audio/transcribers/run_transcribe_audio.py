import shutil
from jet.audio.transcribers.utils import transcribe_audio, segments_to_srt
from jet.file.utils import save_file
from pathlib import Path
from typing import List  # for type hint

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # recreate empty

if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"

    current_info = None
    current_segments = []
    current_chunk_idx = -1  # track which chunk we're collecting

    all_segments: List = []  # ← NEW: collect globally

    for i, (segment, info, chunk_idx) in enumerate(transcribe_audio(
        audio_path,
        model_name="large-v3",
        device="auto",
        compute_type="int8_float32",
        overlap_seconds=8.0,
        chunk_length_seconds=30,
        vad_filter=True,
        word_timestamps=True,
        language="ja",
        task="translate",
        beam_size=5,
        temperature=[0.0],
        repetition_penalty=1.1,
        condition_on_previous_text=True,
    ), 1):
        if current_chunk_idx == -1:
            current_chunk_idx = chunk_idx
            current_info = info

        # Save previous chunk data when chunk_idx changes
        if current_chunk_idx != chunk_idx:
            print(f"\nSaved chunk_{current_chunk_idx:04d} → {len(current_segments)} segments")
            print(f" Language: {current_info.language} (prob: {current_info.language_probability:.3f})\n")

            # Reset for new chunk
            current_chunk_idx = chunk_idx
            current_info = info
            current_segments = []

        # Update current state and accumulate global segments
        current_segments.append(segment)
        all_segments.append(segment)  # collect globally

        chunk_dir = OUTPUT_DIR / f"chunk_{current_chunk_idx:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        save_file(current_info, chunk_dir / "info.json")
        save_file(current_segments, chunk_dir / "segments.json")

        # Save per-chunk SRT
        srt_content = segments_to_srt(current_segments)
        save_file(srt_content, chunk_dir / "subtitles.srt")

        print(f"{i:>3d}. [chunk {chunk_idx}] [{segment.start:.2f} → {segment.end:.2f}] {segment.text}")

    # Save the FINAL chunk after loop ends
    if current_segments:
        final_dir = OUTPUT_DIR / f"chunk_{current_chunk_idx:04d}"
        final_dir.mkdir(parents=True, exist_ok=True)
        save_file(current_info, final_dir / "info.json")
        save_file(current_segments, final_dir / "segments.json")

        srt_content = segments_to_srt(current_segments)
        save_file(srt_content, final_dir / "subtitles.srt")

        print(f"\nSaved final chunk_{current_chunk_idx:04d} → {len(current_segments)} segments")
        print(f" Language: {current_info.language} (prob: {current_info.language_probability:.3f})")

    # Save merged final SRT (all segments)
    if all_segments:
        final_srt_path = OUTPUT_DIR / "subtitles_all.srt"
        save_file(segments_to_srt(all_segments), final_srt_path)
        print(f"\nSaved complete subtitles → {final_srt_path}")