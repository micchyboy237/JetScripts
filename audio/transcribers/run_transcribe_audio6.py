from faster_whisper import WhisperModel
from jet.models.model_registry.transformers.speech_to_text.whisper_model_registry import WhisperModelsType

_model: str | WhisperModelsType = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"
# _model: str | WhisperModelsType = "large-v3"

model = WhisperModel(_model, device="cpu", compute_type="int8")

# Japanese (speech) to English (text) Translation
audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/silero/generated/silero_vad_stream/segment_001/sound.wav"
segments, info = model.transcribe(
    audio_file,
    language="ja",
    task="translate",
    vad_filter=False,
    word_timestamps=True,
    without_timestamps=False,
    condition_on_previous_text=False,
    log_progress=True
)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
