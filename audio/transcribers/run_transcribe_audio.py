import os
import shutil
from faster_whisper import WhisperModel
from datetime import datetime
from pathlib import Path
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# ==============================
# Configuration
# ==============================
audio_path = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav")

model_name = "large-v3"  # or "large-v3-turbo" if you want faster inference

# ==============================
# Model Initialization
# ==============================
logger.info(f"Loading Whisper model: {model_name}")
model = WhisperModel(model_name, device="cpu", compute_type="int8_float32")
logger.info("Model loaded successfully")

# ==============================
# High-Accuracy Japanese → English Translation
# ==============================
logger.info(f"Starting translation of: {audio_path.name}")

segments, info = model.transcribe(
    audio=str(audio_path),
    language="ja",
    task="translate",

    # Decoding: Maximum accuracy
    beam_size=10,
    patience=2.0,
    temperature=0.0,
    length_penalty=1.0,
    best_of=1,
    log_prob_threshold=-0.5,

    # Context & consistency
    condition_on_previous_text=True,

    # Japanese punctuation handling
    prepend_punctuations="\"'“¿([{-『「（［",
    append_punctuations="\"'.。,，!！?？:：”)]}、。」」！？",

    # Clean input
    vad_filter=True,
    vad_parameters=None,

    # Output options
    without_timestamps=False,
    word_timestamps=True,
    chunk_length=30,
    log_progress=True,
)

# ==============================
# Log Key Metadata
# ==============================
logger.info("=" * 60)
logger.info("TRANSCRIPTION / TRANSLATION COMPLETE")
logger.info("=" * 60)
logger.info(f"Detected Language       : {info.language} (probability: {info.language_probability:.3f})")
logger.info(f"Original Duration       : {info.duration:.2f}s")
logger.info("-" * 60)

# ==============================
# Collect & Log Full Translated Text + Timestamps
# ==============================
full_translation = []
all_segments = []
for i, segment in enumerate(segments, start=1):
    text = segment.text.strip()
    start = segment.start
    end = segment.end

    full_translation.append(text)
    all_segments.append(segment)

    logger.info(f"[{i:3d}] {start:6.2f} → {end:6.2f} | {text}")

# Final clean output
final_text = " ".join(full_translation).strip()

logger.info("=" * 60)
logger.info("FINAL ENGLISH TRANSLATION")
logger.info("=" * 60)
logger.info(final_text)
logger.info("=" * 60)

# ==============================
# Optional: Save to file
# ==============================
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"{audio_path.stem}_translated_{timestamp}.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Source File     : {audio_path.name}\n")
    f.write(f"Model           : {model_name}\n")
    f.write("Task            : translate (Japanese → English)\n")
    f.write(f"Processed at    : {datetime.now().isoformat()}\n")
    f.write(f"Duration        : {info.duration:.2f}s\n")
    f.write(f"Segments        : {len(all_segments)}\n")
    f.write("\n" + "="*60 + "\n")
    f.write("FULL TRANSLATION\n")
    f.write("="*60 + "\n")
    f.write(final_text)

save_file(info, f"{OUTPUT_DIR}/info.json")
save_file(all_segments, f"{OUTPUT_DIR}/segments.json")

logger.info(f"Translation saved to: {output_file}")