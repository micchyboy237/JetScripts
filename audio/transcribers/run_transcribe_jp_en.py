import os
import shutil
import platform
import argostranslate.package
import argostranslate.translate
from faster_whisper import WhisperModel
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def setup_argos_model(from_lang: str = "ja", to_lang: str = "en") -> None:
    """
    Modular setup: Updates package index and installs JA-EN model if missing.
    Idempotent—one-time download (~200-300MB).
    """
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_lang and x.to_code == to_lang,
            available_packages
        ),
        None
    )
    if package_to_install:
        argostranslate.package.install_from_path(package_to_install.download())

def translate_text(text: str, from_lang: str = "ja", to_lang: str = "en") -> str:
    """
    Reusable translation function: Offline JA→EN using Argos Translate.
    Returns natural English; handles empty input gracefully.
    """
    if not text.strip():
        return ""
    return argostranslate.translate.translate(text, from_lang, to_lang)

if __name__ == "__main__":
    # One-time model setup (call once; safe to rerun)
    setup_argos_model()

    sound_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_022942.wav"

    # Choose model: "large-v3" for best accuracy, "large-v3-turbo" for speed
    device = "cuda" if platform.system() == "Windows" else "auto"  # Auto-detects Metal on M1
    model = WhisperModel("large-v3-turbo", device=device)

    # Transcribe Japanese audio → Japanese text
    # segments_iter, info = model.transcribe(
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_022942.wav",
    #     language="ja",
    #     task="translate",
    #     beam_size=7,
    #     best_of=5,
    #     temperature=(0.0, 0.2),
    #     vad_filter=True,
    #     vad_parameters=dict(min_silence_duration_ms=700),
    #     prefix=None,
    #     word_timestamps=True,
    #     log_progress=True,
    # )
    segments_iter, info = model.transcribe(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_022942.wav",
        beam_size=1,                   # greedy
        temperature=[0.0],
        vad_filter=True,
        chunk_length=20,
        word_timestamps=False
    )
    segments = [segment for segment in segments_iter]
    japanese_text = " ".join(seg.text for seg in segments)

    # Translate to English (offline with Argos Translate)
    english_text = translate_text(japanese_text)

    print(f"JA Transcript: {japanese_text[:100]}...")  # Preview (optional)
    print(f"EN Translation: {english_text[:100]}...")  # Preview (optional)

    # Save both
    save_file(segments, f"{OUTPUT_DIR}/segments.json")
    save_file(info, f"{OUTPUT_DIR}/info.json")
    save_file(japanese_text, f"{OUTPUT_DIR}/transcript_ja.txt")
    save_file(english_text, f"{OUTPUT_DIR}/transcript_en.txt")
