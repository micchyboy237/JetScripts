from faster_whisper import WhisperModel
from jet.translators.translate_jp_en_ct2 import translate_ja_to_en

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster", device="cpu", compute_type="int8")

# Example usage
AUDIO_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments/segment_002/sound.wav"

# Japanese ASR
segments, info = model.transcribe(
    AUDIO_PATH,
    language="ja",
    chunk_length=30,  # Better context than 15s
    condition_on_previous_text=True,  # Reduces repetitions/hallucinations
    beam_size=5,
    best_of=5,
    temperature=0.0,  # More deterministic
)
for segment in segments:
    text_original = segment.text
    text_en = translate_ja_to_en(text_original)
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]")
    print(f"JP: {segment.text}")
    print(f"EN: {text_en}")
