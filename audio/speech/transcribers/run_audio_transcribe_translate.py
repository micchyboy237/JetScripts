
from jet.audio.transcribers.base import QuantizedModelSizes, transcribe_audio
from jet.translators.base import translate_text

model_size: QuantizedModelSizes = "small"
audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers.wav"

# 3. Transcribe in original language
text_original = transcribe_audio(audio_path)
print("\nTranscription:")
print(text_original)

# 4. Translate to English
text_en = translate_text(
    text_original,
    return_scores=True,
    return_attention=True,
)
print("\nEnglish translation:")
print(text_en)