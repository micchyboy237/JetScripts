import numpy as np
from faster_whisper import WhisperModel
from transformers import pipeline
from silero_vad import VoiceActivityDetector

# Load WhisperX / Faster-Whisper streaming model (production recommended)
model = WhisperModel(
    model_size="large-v2",
    device="cuda"  # CPU also possible; GPU far faster
)

translation = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-ja-en", # japanese→english
    device=0   # GPU
)

vad = VoiceActivityDetector(
    sample_rate=16000,
    threshold=0.5,
)

async def asr_stream(pcm_chunks):
    """
    Async generator that yields transcripts as they're ready.
    """
    buffer: list[np.ndarray] = []
    for chunk in pcm_chunks:
        # VAD: skip silence
        if not vad.is_speech(chunk):
            continue

        buffer.append(chunk)
        if len(buffer) * len(chunk) > 16000:  # ~1 second speech
            data = np.concatenate(buffer)
            # stream decode partial
            segments, _ = model.transcribe(
                audio=data,
                language="ja",
                initial_prompt=None,
                beam_size=5,
                chunk_size=30
            )
            text = " ".join(s.text for s in segments)
            buffer = []
            yield text

async def translate_ja_to_en(text: str) -> str:
    result = translation(text, max_length=512)
    return result[0]["translation_text"]
