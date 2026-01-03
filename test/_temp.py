# realtime_transcriber.py

import queue
import threading
import logging
from typing import List

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

class RealtimeTranscriber:
    """
    Near-real-time microphone transcriber using faster-whisper.
    
    Features:
    - Simple energy-based pre-VAD + built-in Silero VAD (vad_filter=True)
    - Accumulation of voiced segments
    - Optional batch transcription for better GPU utilization
    - Word-level timestamps with rich display
    - Optimized defaults for modest GPUs (GTX 1660)
    """

    def __init__(
        self,
        model_size: str = "distil-large-v3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16",  # "int8" for lower VRAM
        batch_size: int = 4,
        vad_threshold: float = 0.035,
        sample_rate: int = 16000,
        chunk_seconds: float = 1.0,
        silence_seconds: float = 0.7,
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.batched_model = BatchedInferencePipeline(model=self.model)
        self.batch_size = batch_size

        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_seconds)
        self.silence_duration = silence_seconds
        self.vad_threshold = vad_threshold

        self.audio_queue = queue.Queue(maxsize=50)
        self.transcription_queue = queue.Queue()
        self.running = False
        self.speech_buffer = np.array([], dtype=np.float32)
        self.silence_timer = 0.0

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        if status:
            console.print(status)
        audio = indata[:, 0].astype(np.float32)
        self.audio_queue.put(audio.copy())

    def _vad_energy(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD using RMS for better scaling."""
        # Use RMS (root mean square) instead of L2 norm for amplitude-independent detection
        rms = np.sqrt(np.mean(audio**2))
        # Threshold tuned for normalized [-1, 1] float audio (typical speech RMS ~0.05-0.3)
        return rms > 0.02  # Lowered and fixed; adjustable if needed

    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        segments, _ = self.batched_model.transcribe(
            audio_chunk,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=800),
            word_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)

        if console.is_terminal:
            table = Table(title="Word Timestamps")
            table.add_column("Word")
            table.add_column("Start (s)")
            table.add_column("End (s)")
            for seg in segments:
                for word in seg.words:
                    table.add_row(word.word.strip(), f"{word.start:.2f}", f"{word.end:.2f}")
            console.print(table)

        return text

    def _transcribe_batch(self, audio_batch: List[np.ndarray]):
        with tqdm(total=len(audio_batch), desc="Transcribing batch", leave=False) as pbar:
            for audio_chunk in audio_batch:
                text = self._transcribe_chunk(audio_chunk)
                console.print(f"[green]Transcript:[/green] {text}")
                pbar.update(1)
                self.transcription_queue.task_done()

    def _transcription_worker(self):
        batch: List[np.ndarray] = []

        while self.running or not self.transcription_queue.empty():
            try:
                audio_chunk = self.transcription_queue.get(timeout=0.5)
                batch.append(audio_chunk)

                if len(batch) >= self.batch_size:
                    self._transcribe_batch(batch)
                    batch.clear()

            except queue.Empty:
                if batch:
                    self._transcribe_batch(batch)
                    batch.clear()
                continue

    def start(self):
        self.running = True
        threading.Thread(target=self._transcription_worker, daemon=True).start()

        console.print("[bold cyan]Starting real-time transcription... Speak now![/bold cyan]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            while self.running:
                try:
                    chunk = self.audio_queue.get()
                    chunk_float = chunk / 32768.0

                    if self._vad_energy(chunk):
                        self.speech_buffer = np.concatenate((self.speech_buffer, chunk_float))
                        self.silence_timer = 0.0
                    else:
                        self.silence_timer += len(chunk) / self.sample_rate

                        if self.speech_buffer.size > 0 and self.silence_timer >= self.silence_duration:
                            self.transcription_queue.put(self.speech_buffer.copy())
                            self.speech_buffer = np.array([], dtype=np.float32)
                            self.silence_timer = 0.0

                        trailing = chunk_float[-int(0.3 * self.sample_rate):]
                        self.speech_buffer = np.concatenate((self.speech_buffer, trailing))

                except KeyboardInterrupt:
                    break

        self.stop()

    def stop(self):
        self.running = False
        if self.speech_buffer.size > 0:
            self.transcription_queue.put(self.speech_buffer.copy())
        console.print("[bold red]Stopped.[/bold red]")


if __name__ == "__main__":
    transcriber = RealtimeTranscriber(
        # model_size="distil-large-v3",
        # device="cuda",
        # compute_type="float16",
        model_size="small",
        device="cpu",
        compute_type="int8",
    )
    transcriber.start()