import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from queue import Queue
from threading import Thread

class JapaneseToEnglishLiveTranscriber:
    """Live Japanese → English translator using faster-whisper large-v3 + Silero VAD (segmented)."""

    def __init__(
        self,
        model_size: str = "large-v3",          # or "large-v3-turbo" / "medium" for faster
        compute_type: str = "int8_float16",    # int8_float16 is fastest on GTX 1660 with negligible quality loss
        beam_size: int = 5,
        vad_threshold: float = 0.6,          # 0.5–0.7 works well for Japanese
        min_silence_duration_ms: int = 400,     # adjust if too chatty or too slow
    ):
        self.model = WhisperModel(model_size, device="cuda", compute_type=compute_type, download_root="./models")
        self.beam_size = beam_size
        self.samplerate = 16000
        self.block_size = 1024  # ~64 ms chunks, good reactivity
        self.vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
        self.VADIterator = utils[3]  # VADIterator class
        self.vad_iterator = self.VADIterator(self.vad_model, threshold=vad_threshold, sampling_rate=self.samplerate, min_silence_duration_ms=min_silence_duration_ms)
        self.audio_queue = Queue()
        self.current_speech_chunks = []

    def _transcribe_buffer(self, audio: np.ndarray):
        segments, _ = self.model.transcribe(
            audio,
            language="ja",
            task="translate",
            beam_size=self.beam_size,
            temperature=0.0,
            best_of=5,
            patience=1.0,
            compression_ratio_threshold=2.4,  # filter bad segments
            logprob_threshold=-0.5,
            no_speech_threshold=0.6,
            vad_filter=False,  # we already did VAD
        )
        text = "".join(segment.text for segment in segments).strip()
        if text:
            print(text, flush=True)

    def _processing_thread(self):
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:  # poison pill for shutdown
                break

            speech_event = self.vad_iterator(torch.from_numpy(chunk), self.samplerate)

            if speech_event:
                if "start" in speech_event:
                    self.current_speech_chunks = [chunk]
                    print("\r[Speaking...]", end="", flush=True)

                if "end" in speech_event:
                    self.current_speech_chunks.append(chunk)
                    audio_np = np.concatenate(self.current_speech_chunks)
                    self._transcribe_buffer(audio_np)
                    self.current_speech_chunks = []
                    self.vad_iterator.reset_states()  # important: fresh state for next utterance
                    print("\r" + " " * 80 + "\r", end="", flush=True)  # clear line

            else:
                if self.current_speech_chunks:
                    self.current_speech_chunks.append(chunk)

    def audio_callback(self, indata: np.ndarray, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy()[:, 0])  # mono float32

    def run(self):
        print("Japanese → English live translation ready (Ctrl+C to stop)")
        with sd.InputStream(samplerate=self.samplerate, channels=1, dtype="float32", blocksize=self.block_size, callback=self.audio_callback):
            thread = Thread(target=self._processing_thread, daemon=True)
            thread.start()
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print("\nStopped.")
                self.audio_queue.put(None)
                thread.join()

if __name__ == "__main__":
    model_size = "small"
    compute_type = "int8"

    transcriber = JapaneseToEnglishLiveTranscriber(
        model_size=model_size,
        compute_type=compute_type,
    )
    transcriber.run()