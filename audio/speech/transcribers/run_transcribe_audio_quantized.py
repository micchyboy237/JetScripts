
from jet.audio.transcribers.base import QuantizedModelSizes, transcribe_audio
from jet.translators.translate_jp_en2 import translate_ja_to_en

model_size: QuantizedModelSizes = "small"
audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav"

# 3. Transcribe in original language
text_original = transcribe_audio(audio_path)
print("\nTranscription:")
print(text_original)

# 4. Translate to English
text_en = translate_ja_to_en(
    text_original,
    return_scores=True,
    return_attention=True,
)
print("\nEnglish translation:")
print(text_en)