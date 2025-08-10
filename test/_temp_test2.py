# src/recorder.py
from typing import Optional
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

def record_to_wav(
    out_path: str,
    duration_seconds: float,
    samplerate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Record audio from default microphone and save to a WAV file (16-bit PCM).
    Returns the path to the saved file.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = int(duration_seconds * samplerate)
    # sounddevice returns float32 in range [-1,1]
    recording: np.ndarray = sd.rec(frames, samplerate=samplerate, channels=channels, dtype="float32")
    sd.wait()
    # Convert float32 [-1,1] to 16-bit PCM when writing via soundfile (it handles float to PCM)
    sf.write(str(out), recording, samplerate, subtype="PCM_16")
    return str(out)


# src/transcriber.py
from typing import Protocol
from pathlib import Path

class Transcriber(Protocol):
    def transcribe(self, wav_path: str) -> str:
        """Transcribe audio from a wav file and return the text."""
        ...

# -------- Whisper implementation (optional, higher quality) ----------
try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore

class WhisperTranscriber:
    def __init__(self, model_name: str = "base"):
        if whisper is None:
            raise RuntimeError("whisper package is not installed. Install via `pip install -U openai-whisper` or from GitHub.")
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

    def transcribe(self, wav_path: str) -> str:
        self._ensure_model()
        path = str(Path(wav_path))
        # whisper returns dict with 'text'
        result = self._model.transcribe(path)
        return result.get("text", "").strip()

# ---------- Vosk implementation (offline) ----------
try:
    from vosk import Model, KaldiRecognizer  # type: ignore
    import json
    import wave
except Exception:
    Model = None  # type: ignore

class VoskTranscriber:
    def __init__(self, model_path: str):
        if Model is None:
            raise RuntimeError("vosk not installed. pip install vosk and download a model.")
        self.model = Model(model_path)

    def transcribe(self, wav_path: str) -> str:
        # expects WAV PCM mono
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1:
            raise ValueError("VoskTranscriber expects mono WAV input.")
        rec = KaldiRecognizer(self.model, wf.getframerate())
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                j = json.loads(rec.Result())
                results.append(j.get("text", ""))
        # final
        final = json.loads(rec.FinalResult())
        results.append(final.get("text", ""))
        wf.close()
        return " ".join(part for part in results if part).strip()


# examples/run_transcription.py
from pathlib import Path
# from src.recorder import record_to_wav
# from src.transcriber import WhisperTranscriber, VoskTranscriber

OUT = Path("recordings")
OUT.mkdir(exist_ok=True)

def main_whisper():
    wav = OUT / "sample_whisper.wav"
    print("Recording 5 seconds...")
    record_to_wav(str(wav), duration_seconds=5.0)
    print("Transcribing with Whisper...")
    t = WhisperTranscriber(model_name="small")  # choose model
    text = t.transcribe(str(wav))
    print("Transcription:", text)

def main_vosk(model_path: str):
    wav = OUT / "sample_vosk.wav"
    print("Recording 5 seconds...")
    record_to_wav(str(wav), duration_seconds=5.0)
    print("Transcribing with Vosk...")
    t = VoskTranscriber(model_path)
    text = t.transcribe(str(wav))
    print("Transcription:", text)

if __name__ == "__main__":
    # choose which backend to run
    # main_whisper()
    # or
    # main_vosk("/path/to/vosk-model-small-en-us")
    pass
