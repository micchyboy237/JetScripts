import os
import shutil
from faster_whisper import WhisperModel
from jet.audio.utils import split_audio, merge_audio_chunks

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_211631.wav"
    
    # Load model (GPU example; use device="cpu" for CPU)
    model = WhisperModel("large-v3", device="cpu")
    segment_duration: float = 10.0
    overlap_duration: float = 2.0
    sample_rate: int = 16000

    chunks = split_audio(audio_path, segment_duration=segment_duration, overlap_duration=overlap_duration, sample_rate=sample_rate)

    chunk_binaries = [chunk[0] for chunk in chunks]
    merge_audio_chunks(chunk_binaries, output_path=f"{OUTPUT_DIR}/merged_output.wav", overlap_duration=overlap_duration, sample_rate=sample_rate)
