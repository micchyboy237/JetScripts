# faster_whisper_custom/transcribe.py
from __future__ import annotations

import os
import shutil
import gc
import platform
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Literal, Optional, Tuple, Union, overload

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import Segment, TranscriptionInfo

from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

DeviceType = Literal["cpu", "cuda", "mps", "auto"]


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for custom long-form chunking with overlap."""
    chunk_length_seconds: int = 10
    overlap_seconds: float = 2.0
    vad_filter: bool = True
    vad_parameters: Optional[dict] = None


@dataclass(frozen=True)
class TranscriptionOptions:
    """All optional transcribe() arguments you want to expose."""
    language: Optional[str] = None
    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: Union[float, List[float], Tuple[float, ...]] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )
    word_timestamps: bool = True
    hotwords: Optional[Union[str, List[str]]] = None
    log_progress: bool = False
    # Add any other frequent args here as needed


class CustomOverlapTranscriber:
    """
    Generic, reusable transcriber with full control over chunk overlap.
    Works reliably on Mac M1/M2 (Apple Silicon), Windows CUDA, and CPU.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: DeviceType = "auto",
        compute_type: str = "float16",
    ):
        """
        Initialise WhisperModel with proper device/compute_type mapping.
        faster-whisper (ctranslate2 backend) does NOT support native "mps".
        We map:
          - "mps" or Apple Silicon + auto → cpu + int8 (fastest real option)
          - CUDA available → cuda + float16
          - fallback → cpu + int8
        """
        # Resolve final device
        if device == "auto":
            if torch.cuda.is_available():
                final_device = "cuda"
            elif platform.system() == "Darwin" and torch.backends.mps.is_available():
                final_device = "cpu"          # Apple Silicon: use int8 for speed
            else:
                final_device = "cpu"
        else:
            # Explicit "mps" → force cpu (ctranslate2 doesn't support mps)
            final_device = "cpu" if device == "mps" else device

        # Resolve compute_type that ctranslate2 actually supports
        if final_device == "cuda":
            final_compute = "float16" if compute_type != "float32" else "float32"
        else:
            # CPU path (including Apple Silicon): int8 is significantly faster
            final_compute = "int8" if compute_type != "float32" else "float32"

        self.device_used = final_device
        self.compute_type_used = final_compute

        print(f"Loading Whisper model '{model_name}' on {final_device} with {final_compute}...")

        self.model = WhisperModel(
            model_name,
            device=final_device,
            compute_type=final_compute,
        )

    @overload
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        chunking: ChunkingConfig = ...,
        options: TranscriptionOptions = ...,
        return_info: Literal[True] = True,
    ) -> Tuple[Generator[Segment, None, None], TranscriptionInfo]: ...

    @overload
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        chunking: ChunkingConfig = ...,
        options: TranscriptionOptions = ...,
        return_info: Literal[False] = ...,
    ) -> Generator[Segment, None, None]: ...

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        chunking: ChunkingConfig = ChunkingConfig(),
        options: TranscriptionOptions = TranscriptionOptions(),
        return_info: bool = True,
    ) -> Union[
        Generator[Segment, None, None],
        Tuple[Generator[Segment, None, None], TranscriptionInfo],
    ]:
        """Transcribe audio with custom chunk overlap. Accepts path or np.ndarray."""
        # Load audio once
        if isinstance(audio, (str, Path)):
            audio_array = decode_audio(str(audio), sampling_rate=16000)
        else:
            audio_array = np.asarray(audio, dtype=np.float32)
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=0)  # stereo → mono

        sample_rate = 16000
        chunk_samples = chunking.chunk_length_seconds * sample_rate
        overlap_samples = int(chunking.overlap_seconds * sample_rate)

        all_segments: List[Segment] = []
        info: Optional[TranscriptionInfo] = None

        start_sample = 0
        while start_sample < len(audio_array):
            end_sample = start_sample + chunk_samples
            chunk_audio = audio_array[start_sample:end_sample]

            segs, chunk_info = self.model.transcribe(
                chunk_audio,
                language=options.language,
                task=options.task,
                beam_size=options.beam_size,
                best_of=options.best_of,
                patience=options.patience,
                temperature=options.temperature,
                word_timestamps=options.word_timestamps,
                log_progress=options.log_progress,
                vad_filter=chunking.vad_filter,
                vad_parameters=chunking.vad_parameters,
                hotwords=(
                    options.hotwords
                    if isinstance(options.hotwords, str)
                    else ",".join(options.hotwords or [])
                ),
            )

            offset_seconds = start_sample / sample_rate
            for seg in segs:
                adjusted = self._offset_segment(seg, offset_seconds)
                all_segments.append(adjusted)

            if info is None:
                info = chunk_info

            start_sample += chunk_samples - overlap_samples

        def segment_generator() -> Generator[Segment, None, None]:
            yield from all_segments

        final_info = info or TranscriptionInfo(
            language="en", language_probability=1.0, duration=len(audio_array) / sample_rate
        )

        return (segment_generator(), final_info) if return_info else segment_generator()

    @staticmethod
    def _offset_segment(seg: Segment, offset: float) -> Segment:
        """Create new Segment with correct global timestamps (including words)."""
        if not seg.words:
            return Segment(
                id=seg.id,
                seek=seg.seek,
                start=seg.start + offset,
                end=seg.end + offset,
                text=seg.text,
                tokens=seg.tokens,
                temperature=seg.temperature,
                avg_logprob=seg.avg_logprob,
                compression_ratio=seg.compression_ratio,
                no_speech_prob=seg.no_speech_prob,
                words=[],
            )

        adjusted_words = [
            type("Word", (), {
                "start": (w.start + offset) if w.start is not None else None,
                "end": (w.end + offset) if w.end is not None else None,
                "word": w.word,
                "probability": w.probability,
            })
            for w in seg.words
        ]

        return Segment(
            id=seg.id,
            seek=seg.seek,
            start=seg.start + offset,
            end=seg.end + offset,
            text=seg.text,
            tokens=seg.tokens,
            temperature=seg.temperature,
            avg_logprob=seg.avg_logprob,
            compression_ratio=seg.compression_ratio,
            no_speech_prob=seg.no_speech_prob,
            words=adjusted_words,
        )

    def __del__(self):
        """Safely release the underlying model and free memory."""
        try:
            if hasattr(self, "model"):
                del self.model
            gc.collect()
        except Exception:
            pass


import argostranslate.package
import argostranslate.translate

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


# Example usage
if __name__ == "__main__":
    transcriber = CustomOverlapTranscriber(
        model_name="large-v3",
        device="auto",           # Works perfectly on M1, CUDA, or CPU
        compute_type="float16"
    )

    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_022942.wav"

    segments_iter, info = transcriber.transcribe(
        audio_path,
        chunking=ChunkingConfig(
            chunk_length_seconds=30,
            overlap_seconds=10.0,
            vad_filter=True,
        ),
        options=TranscriptionOptions(
            beam_size=5,
            log_progress=True
        ),
        return_info=True,
    )

    print(f"Detected Language: {info.language} | Duration: {info.duration/60:.1f} min")
    segments = []
    for segment in segments_iter:
        segments.append(segment)
        print(f"[{segment.start:.2f} → {segment.end:.2f}] {segment.text}")

    # One-time model setup (call once; safe to rerun)
    setup_argos_model()

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