from pathlib import Path
from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps
from jet.file.utils import save_file
from rich.console import Console
import torchaudio
from silero_vad.utils_vad import read_audio
import os
import shutil

console = Console()

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR = Path(OUTPUT_DIR) / "segments"
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")

    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.5,
        sampling_rate=16000,
        return_seconds=True,
        time_resolution=2,
    )

    # Load full waveform once for efficient slicing
    waveform = read_audio(audio_file, sampling_rate=16000)
    # read_audio returns 1D tensor → make it (1, samples) for torchaudio
    waveform = waveform.unsqueeze(0)

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
        )

        # Create safe folder name
        folder_name = (
            f"{seg['idx']:03d}_"
            f"{seg['start']:.2f}_"
            f"{seg['end']:.2f}_"
            f"{seg['duration']:.3f}_"
            f"{seg['prob']:.3f}"
        )
        seg_dir = SEGMENTS_DIR / folder_name
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Save individual segment metadata
        save_file(seg, seg_dir / "segment.json")

        # Extract and save audio slice
        start_sample = int(seg['start'] * 16000)
        end_sample = int(seg['end'] * 16000)
        segment_audio = waveform[:, start_sample:end_sample]  # (1, N) – now safe
        torchaudio.save(
            str(seg_dir / "sound.wav"),
            segment_audio,
            sample_rate=16000,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        console.print(f"   → Saved to [bold blue]{seg_dir.relative_to(OUTPUT_DIR)}[/bold blue]")

    save_file(segments, f"{OUTPUT_DIR}/speech_timestamps.json")