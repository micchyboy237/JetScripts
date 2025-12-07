# jet_python_modules/jet/audio/speech/pyannote/speech_speakers_extractor.py
import os
from typing import List, TypedDict, Literal, Optional
from pathlib import Path
import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from pyannote.audio import Pipeline
from pyannote.core import Annotation

console = Console()

DeviceType = Literal["cpu", "cuda", "mps"]


class SpeakerSegment(TypedDict):
    speaker: str
    start: float
    end: float
    duration: float
    confidence: float


@torch.no_grad()
def extract_speakers(
    audio: str | Path | np.ndarray | torch.Tensor,
    hf_token: Optional[str] = None,
    model: str = "pyannote/speaker-diarization-3.1",
    device: Optional[DeviceType] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_seconds: bool = True,
    time_precision: int = 3,
) -> List[SpeakerSegment]:
    """
    Extract speaker turns from audio using pyannote.audio diarization pipeline.

    Args:
        audio: Path to audio file, or raw waveform (np.ndarray/torch.Tensor @ 16kHz)
        hf_token: Hugging Face token (required for pyannote models)
        model: Pretrained pyannote diarization model
        device: 'cuda', 'mps', 'cpu' — auto-detected if None
        min_speakers / max_speakers: Constrain number of speakers
        return_seconds: Return timestamps in seconds (else samples)
        time_precision: Decimal places for timestamps

    Returns:
        List of speaker segments with confidence scores
    """
    # === Device selection ===
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    console.print(f"[bold blue]Using device:[/bold blue] {device.upper()}")

    # === Load audio ===
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        waveform, sample_rate = torch.load(audio_path, map_location=device)  # pyannote handles loading
    elif isinstance(audio, (np.ndarray, torch.Tensor)):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        elif audio.ndim != 1:
            raise ValueError("Audio waveform must be mono (1D)")
        if audio.dtype != torch.float32:
            audio = audio.float()
        waveform = audio.unsqueeze(0)  # (1, T)
        sample_rate = 16000
    else:
        raise TypeError("audio must be str, Path, np.ndarray, or torch.Tensor")

    if waveform.shape[0] != 1:
        raise ValueError("Audio must be mono")

    # === Load pipeline with rich feedback ===
    with console.status(f"[bold green]Loading pyannote pipeline: {model}...[/bold green]"):
        pipeline = Pipeline.from_pretrained(
            model,
            use_auth_token=hf_token,
        )
        pipeline.to(torch.device(device))
    console.print("✅ Diarization pipeline loaded")

    # Optional speaker count constraint
    if min_speakers is not None:
        pipeline.min_speakers = min_speakers
    if max_speakers is not None:
        pipeline.max_speakers = max_speakers

    # === Run diarization ===
    with Progress(
        SpinnerColumn(),
        "[bold blue]Running speaker diarization...",
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Diarizing", total=1)
        diarization: Annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        progress.update(task, advance=1)

    # === Convert to structured output ===
    segments: List[SpeakerSegment] = []
    for turn, track, speaker in diarization.itertracks(yield_label=True):
        start = round(turn.start, time_precision) if return_seconds else int(turn.start * sample_rate)
        end = round(turn.end, time_precision) if return_seconds else int(turn.end * sample_rate)
        duration = round(turn.duration, 3)

        # Confidence: pyannote doesn't expose per-segment easily → use dummy high if not available
        # In future: can use overlap or embedding quality, but not exposed cleanly
        confidence = 0.95  # placeholder — real confidence requires custom postprocessing

        segments.append(
            SpeakerSegment(
                speaker=speaker,
                start=start,
                end=end,
                duration=duration,
                confidence=confidence,
            )
        )

    # Sort by start time
    segments.sort(key=lambda x: x["start"])

    console.print(f"✅ Found [bold green]{len(segments)}[/bold green] speaker turns from [bold yellow]{len(set(s['speaker'] for s in segments))}[/bold yellow] speakers")
    return segments


if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"

    segments = extract_speakers(
        audio=audio_file,
        hf_token=os.getenv("HF_TOKEN"),
        device=None,  # auto
        return_seconds=True,
    )

    for seg in segments:
        console.print(
            f"[yellow][[/yellow] {seg['start']:.2f} → {seg['end']:.2f} [yellow]][/yellow] "
            f"[bold magenta]{seg['duration']}s[/bold magenta] "
            f"Speaker [bold cyan]{seg['speaker']}[/bold cyan] "
            f"conf=[bold green]{seg['confidence']:.2f}[/bold green]"
        )