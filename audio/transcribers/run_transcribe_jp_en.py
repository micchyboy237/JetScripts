import os
import shutil
import dataclasses
from faster_whisper import WhisperModel
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

sound_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_211631.wav"
# Load model (GPU example; use device="cpu" for CPU)
model = WhisperModel("large-v3", device="cpu")

# Transcribe Japanese audio and translate to English
segments_iter, info = model.transcribe(
    sound_file,  # Your audio file (supports MP3, WAV, M4A, etc.)
    beam_size=1,           # Improves accuracy (default: 5)
    temperature=[0.0],
    language="ja",         # Source language: Japanese
    task="translate",      # Output: English translation
    condition_on_previous_text=False,  # Avoid carry-over errors
    word_timestamps=True,  # Optional: Get word-level timestamps
    vad_filter=True        # Optional: Filter out silence using Silero VAD
)

save_file(info, f"{OUTPUT_DIR}/info.json")

print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
segments = []
for segment in segments_iter:
    segment_dict = dataclasses.asdict(segment)
    segments.append(segment_dict)
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

save_file(segments, f"{OUTPUT_DIR}/segments.json")