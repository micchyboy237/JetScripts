import os
import shutil
from jet.file.utils import save_file
from jet.audio.transcribers.audio_context_transcriber import AudioContextTranscriber

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    transcriber_context = AudioContextTranscriber(model_size=model_size, sample_rate=None)

    # Transcribe Japanese audio → Japanese text
    segments, info = model.transcribe(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_211631.wav",
        language="ja",
        task="translate",
        beam_size=7,
        best_of=5,
        temperature=(0.0, 0.2),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700),
        prefix=None,
        word_timestamps=True,
    )
    english_text = " ".join(seg.text for seg in segments)

    # Save both
    save_file(english_text, f"{OUTPUT_DIR}/transcript_en.txt")

    print(f"EN Translation: {english_text[:100]}...")  # Preview (optional)