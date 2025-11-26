import os
import shutil
from faster_whisper import WhisperModel
from jet.audio.utils import save_audio_chunks, split_audio

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_022942.wav"
    # Load model (GPU example; use device="cpu" for CPU)
    model = WhisperModel("large-v3", device="cpu")
    segment_duration: float = 10.0
    overlap_duration: float = 1.0
    sample_rate: int = 16000

    # for segment, start_time, end_time in split_audio(sound_file, segment_duration, overlap_duration, sample_rate):
    #     if len(segment) > 0:  # Skip empty segments
    #         segments, info = transcribe_audio(
    #             segment, model)

    chunks = split_audio(audio_path, segment_duration=segment_duration, overlap_duration=overlap_duration, sample_rate=sample_rate)

    saved_files = save_audio_chunks(
        chunks,
        output_dir=f"{OUTPUT_DIR}/output_chunks",
        prefix="recording",
        format="wav",
        sample_rate=16000
    )

    print(f"Saved {len(saved_files)} chunks:")
    for p in saved_files:
        print("  →", p.name)