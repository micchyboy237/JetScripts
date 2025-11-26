import os
import shutil
import numpy as np
from faster_whisper import WhisperModel
from jet.audio.utils import load_audio, split_audio, merge_in_memory_chunks
import soundfile as sf

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    
    # Load model (GPU example; use device="cpu" for CPU)
    model = WhisperModel("large-v3", device="cpu")
    segment_duration: float = 10.0
    overlap_duration: float = 2.0
    sample_rate: int = 16000

    audio, sr = load_audio(audio_path)

    # for segment, start_time, end_time in split_audio(sound_file, segment_duration, overlap_duration, sample_rate):
    #     if len(segment) > 0:  # Skip empty segments
    #         segments, info = transcribe_audio(
    #             segment, model)

    chunks = split_audio(audio_path, segment_duration=segment_duration, overlap_duration=overlap_duration, sample_rate=sample_rate)

    reconstructed_audio = merge_in_memory_chunks(chunks, overlap_duration, sample_rate)

    print(f"Reconstructed audio: {len(reconstructed_audio)/sr:.2f}s")

    # 5. Save and compare
    output_path = f"{OUTPUT_DIR}/reconstructed_from_chunks.wav"
    sf.write(output_path, reconstructed_audio, samplerate=sr, subtype="PCM_16")
    print(f"Saved reconstructed audio → {output_path}")

    # 6. Verify it's bit-perfect with original (except possible float rounding)
    max_diff = np.max(np.abs(audio.astype(np.float32) - reconstructed_audio.astype(np.float32)))
    print(f"Max sample difference vs original: {max_diff:.10f} → {'Perfect!' if max_diff < 1e-6 else 'Minor float diff (normal)'}" )