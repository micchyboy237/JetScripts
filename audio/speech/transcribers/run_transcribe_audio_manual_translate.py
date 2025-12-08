from pathlib import Path

from jet.audio.transcribers.base import QuantizedModelSizes, detect_language, load_audio, load_whisper_ct2_model, preprocess_audio, transcribe
from jet.translators.translate_jp_en2 import translate_ja_to_en

model_size: QuantizedModelSizes = "small"
model_dir = Path("~/.cache/hf_ctranslate2_models").expanduser() / f"whisper-{model_size}-ct2"
audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav"

# 1. Load once
model, processor = load_whisper_ct2_model(model_size, str(model_dir))
audio = load_audio(audio_path)
features = preprocess_audio(audio, processor)

# 2. Detect language (optional but recommended for transcription)
lang_token, prob = detect_language(model, features)
print(f"Detected: {lang_token} ({prob:.2%})")

# 3. Transcribe in original language
text_original = transcribe(model, features, processor, language_token=lang_token)
print("\nTranscription:")
print(text_original)

# 4. Translate to English
text_en = translate_ja_to_en(text_original)
print("\nEnglish translation:")
print(text_en)