# File: ja_to_en_translator.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict, overload

import torch
from datasets import Audio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from transformers import pipeline

console = Console()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

ASR_MODEL = "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all"   # kotoba-whisper-v2.0
TRANSLATOR_MODEL = "facebook/nllb-200-distilled-600M"  # 600M, fast & accurate


# --------------------------------------------------------------------------- #
# Typed result
# --------------------------------------------------------------------------- #
class TranslationResult(TypedDict):
    english_text: str
    japanese_text: str
    duration_seconds: float
    processing_time_seconds: float


# --------------------------------------------------------------------------- #
# Reusable translator class
# --------------------------------------------------------------------------- #
class JapaneseToEnglishTranslator:
    def __init__(self) -> None:
        console.log(f"[bold blue]Loading models on {DEVICE.upper()}...[/bold blue]")

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_asr = progress.add_task("[cyan]Loading ASR model...[/cyan]", total=None)
            task_trans = progress.add_task("[magenta]Loading Translator model...[/magenta]", total=None)

            # Load ASR model
            self.asr = pipeline(
                "automatic-speech-recognition",
                model=ASR_MODEL,
                torch_dtype=DTYPE,
                device=DEVICE,
                chunk_length_s=30,
            )
            progress.update(task_asr, completed=True, visible=False)

            # Load Translator model
            self.translator = pipeline(
                "translation",
                model=TRANSLATOR_MODEL,
                torch_dtype=DTYPE,
                device=DEVICE,
                src_lang="jpn_Jpan",
                tgt_lang="eng_Latn",
                max_length=512,
            )
            progress.update(task_trans, completed=True, visible=False)

        console.log("[bold green]Both ASR and Translator models ready![/bold green]")

    @torch.inference_mode()
    def transcribe_and_translate(
        self,
        audio_path: str | Path,
        *,
        return_japanese_text: bool = True,
    ) -> TranslationResult:
        audio_path = Path(audio_path)

        # Load & resample once
        audio = Audio(sampling_rate=16_000)
        waveform = audio.decode_example(audio.encode_example(str(audio_path)))["array"]

        start_time = torch.cuda.Event(enable_timing=True) if DEVICE == "cuda" else None
        if start_time:
            start_time.record()

        # Step 1: Japanese ASR (fast distilled Whisper)
        asr_result = self.asr(str(audio_path), return_timestamps=False)
        japanese_text: str = asr_result["text"].strip()

        # Step 2: Translation (NLLB distilled)
        translation = self.translator(japanese_text, max_length=512)[0]["translation_text"]
        english_text: str = translation.strip()

        processing_time = (
            start_time.elapsed_time(torch.cuda.Event(enable_timing=True)) / 1000
            if start_time
            else None
        )

        result: TranslationResult = {
            "english_text": english_text,
            "japanese_text": japanese_text if return_japanese_text else "",
            "duration_seconds": len(waveform) / 16_000,
            "processing_time_seconds": processing_time or 0.0,
        }

        return result

    # Convenience overloads
    @overload
    def __call__(self, audio_path: str | Path) -> str: ...
    @overload
    def __call__(self, audio_path: str | Path, detailed: Literal[True]) -> TranslationResult: ...

    def __call__(
        self,
        audio_path: str | Path,
        detailed: bool = False,
    ) -> str | TranslationResult:
        result = self.transcribe_and_translate(audio_path)
        return result if detailed else result["english_text"]


# --------------------------------------------------------------------------- #
# Global singleton (import & reuse anywhere)
# --------------------------------------------------------------------------- #
translator = JapaneseToEnglishTranslator()


# --------------------------------------------------------------------------- #
# Simple CLI example (run directly)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from argparse import ArgumentParser

    DEFAULT_AUDIO = Path("C:/Users/druiv/Desktop/Jet_Files/Jet_Windows_Workspace/python_scripts/samples/audio/data/1.wav")

    parser = ArgumentParser(
        description="Japanese → English speech translator",
        epilog="If no audio file is provided, defaults to the sample file."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",                                          # makes it optional
        type=Path,
        default=DEFAULT_AUDIO,
        help="Path to input .wav file (optional — uses default sample if omitted)",
    )
    parser.add_argument(
        "--no-detailed",
        dest="detailed",
        action="store_false",
        help="Disable detailed segment output",
    )
    parser.set_defaults(detailed=True)

    args = parser.parse_args()
    audio_path: Path = args.audio_file.expanduser().resolve()

    if not audio_path.is_file():
        console.print(f"[red]Error: File not found → {audio_path}[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Translating:[/bold] {audio_path}")

    result = translator(str(audio_path), detailed=args.detailed)

    rt_factor = result["processing_time_seconds"] / max(result["duration_seconds"], 0.1)

    console.rule("Result")
    console.print(f"[green]English:[/green] {result['english_text']}")
    console.print(f"[blue]Japanese (ASR):[/blue] {result['japanese_text']}")
    console.print(
        f"[bold cyan]Speed:[/bold cyan] {rt_factor:.2f}x real-time "
        f"({result['processing_time_seconds']:.2f}s for {result['duration_seconds']:.1f}s audio)"
    )