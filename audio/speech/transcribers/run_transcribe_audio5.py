# asr_inference.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from jet.file.utils import save_file
import os
import shutil

from jet.translators.google_translator import translate_text

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

console = Console()

# Constants (easy to override if needed)
LANG_ID = "ja"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
TARGET_SAMPLE_RATE = 16_000


class JapaneseWav2Vec2Inference:
    """Reusable Japanese ASR inference using wav2vec2-large-xlsr-53-japanese."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,  # ← This prevents downloading
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "hub"

        console.log(f"Loading model [cyan]{model_id}[/] from local cache only → [green]{self.device}[/]")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ).to(self.device)
        except OSError as e:
            if local_files_only:
                console.print(
                    "[red]Model not found in local cache and `local_files_only=True`.[/]\n"
                    "To download it once, run with `--download` or set local_files_only=False temporarily."
                )
                raise RuntimeError(f"Model {model_id} not found locally. Download it first.") from e

        self.model.eval()

        # Silence deprecated gradient_checkpointing warning (harmless)
        self.model.config.gradient_checkpointing = False

    @torch.no_grad()
    def transcribe(
        self,
        audio: str | Path | Sequence[float] | list[str | Path | Sequence[float]],
    ) -> list[str]:
        """
        Transcribe one or many Japanese audio sources.
        """
        if not isinstance(audio, (list, tuple)):
            audio = [audio]

        # Keep raw NumPy arrays (1D) — DO NOT convert to torch yet
        arrays: list[np.ndarray] = []
        for item in tqdm(audio, desc="Loading audio", leave=False):
            if isinstance(item, (str, Path)):
                import soundfile as sf
                array, sr = sf.read(Path(item), dtype="float32")
                if sr != TARGET_SAMPLE_RATE:
                    raise ValueError(f"Audio {item} must be {TARGET_SAMPLE_RATE} Hz, got {sr} Hz")
            else:
                import numpy as np
                array = np.asarray(item, dtype="float32").squeeze()  # Ensure 1D

            if array.ndim != 1:
                raise ValueError(f"Audio must be 1D, got shape {array.shape}")
            if len(array) == 0:
                raise ValueError("Empty audio array detected")

            arrays.append(array)

        # Let the processor handle resampling, normalization, tensor conversion, padding
        inputs = self.processor(
            arrays,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Forward pass
        logits = self.model(
            input_values=inputs.input_values,
            attention_mask=inputs.attention_mask,
        ).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)
        return [t.strip().upper() for t in transcriptions]  # XLSR models often output uppercase

    def transcribe_and_display(
        self,
        audio_paths: list[str | Path],
        references: list[str] | None = None,
    ) -> None:
        """Convenient method to transcribe and pretty-print results with optional references."""
        predictions = self.transcribe(audio_paths)

        table = Table(title="Japanese ASR Results (wav2vec2-large-xlsr-53-japanese)", show_header=True)
        table.add_column("Index", style="dim")
        if references:
            table.add_column("Reference", style="green")
        table.add_column("Prediction", style="cyan")

        for i, pred in enumerate(predictions):
            row = [str(i)]
            if references:
                row.append(references[i] if i < len(references) else "-")
            row.append(pred)
            table.add_row(*row)

        console.print(table)


# Example usage (can be placed under if __name__ == "__main__"):
if __name__ == "__main__":
    inferencer = JapaneseWav2Vec2Inference()

    # Example: replace dataset with your own local audio files
    # sample_audio_files = [
    #     "samples/japanese_001.wav",
    #     "samples/japanese_002.wav",
    #     # ... add more
    # ]
    # Example: replace dataset with your own local audio files
    sample_audio_files = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/silero/generated/silero_vad_stream/segment_001/sound.wav",
    ]

    # Optional: load reference transcriptions from a file or list
    # sample_references = [
    #     "こんにちは、世界",
    #     "今日の天気は晴れです",
    # ]
    # sample_references = None

    # inferencer.transcribe_and_display(sample_audio_files, references=sample_references)

    ja_predictions = inferencer.transcribe(sample_audio_files)

    console.log("Translating predictions...")
    predictions = []
    for idx, ja_text in tqdm(enumerate(ja_predictions, start=1), "Translating to EN"):
        en_text = translate_text(ja_text)
        predictions.append({
            "num": idx + 1,
            "ja": ja_text,
            "en": en_text,
        })
    
    save_file(predictions, f"{OUTPUT_DIR}/predictions.json")
    