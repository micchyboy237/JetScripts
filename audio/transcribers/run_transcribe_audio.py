import shutil
from jet.audio.transcribers.utils import transcribe_audio
from jet.file.utils import save_file
# === ADD: import pathlib for safe directory creation ===
from pathlib import Path

# === UPDATE: OUTPUT_DIR setup with Path ===
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # recreate empty

if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_022942.wav"

    current_info = None
    current_segments = []
    current_chunk_idx = -1  # track which chunk we're collecting

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
        temperature=[0.0, 0.2, 0.4],
        repetition_penalty=1.1,
        condition_on_previous_text=True,
    ), 1):

        # === CHANGED: Save previous chunk's data when chunk_idx changes ===
        if current_chunk_idx != chunk_idx and current_chunk_idx != -1:
            chunk_dir = OUTPUT_DIR / f"chunk_{current_chunk_idx}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            save_file(current_info, chunk_dir / "info.json")
            save_file(current_segments, chunk_dir / "segments.json")

            print(f"\nSaved chunk_{current_chunk_idx} → {len(current_segments)} segments")
            print(f"   Language: {current_info.language} (prob: {current_info.language_probability:.3f})\n")

            # Reset for new chunk
            current_segments = []
            current_info = info

        # === UPDATE current state ===
        current_chunk_idx = chunk_idx
        current_info = info
        current_segments.append(segment)

        print(f"{i:>3d}. [chunk {chunk_idx}] [{segment.start:.2f} → {segment.end:.2f}] {segment.text}")

    # === ADD: Save the FINAL chunk after loop ends ===
    if current_segments:
        final_dir = OUTPUT_DIR / f"chunk_{current_chunk_idx}"
        final_dir.mkdir(parents=True, exist_ok=True)
        save_file(current_info, final_dir / "info.json")
        save_file(current_segments, final_dir / "segments.json")
        print(f"\nSaved final chunk_{current_chunk_idx} → {len(current_segments)} segments")
        print(f"   Language: {current_info.language} (prob: {current_info.language_probability:.3f})")