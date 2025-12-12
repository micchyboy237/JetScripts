# run_stream_speech_timestamps.py (updated)
import sounddevice as sd
import torch
import numpy as np
from pathlib import Path
import shutil
from rich.console import Console

from jet.audio.speech.silero.stream_speech_timestamps import StreamSpeechTimestampExtractor

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_loopback_device() -> int:
    """Select the default WASAPI loopback device on Windows (cross-platform fallback to default input)."""
    devices = sd.query_devices()
    # Try to find a device with 'loopback' in its name (Windows), fallback to first input
    for i, dev in enumerate(devices):
        try:
            name = dev["name"].lower()
            if "loopback" in name:
                return i
        except Exception:
            pass
    # No loopback device found; fallback to system default input
    console.print("[bold yellow]No explicit loopback device found – using default input (may be microphone on macOS)[/bold yellow]")
    device = sd.default.device
    # sd.default.device may be (input_id, output_id) or just input_id
    if isinstance(device, (list, tuple)):
        return device[0]
    return device

def main():
    sampling_rate = 16000
    window_size_samples = 512
    chunk_duration_s = window_size_samples / sampling_rate

    extractor = StreamSpeechTimestampExtractor(sampling_rate=sampling_rate)

    device = get_loopback_device()
    try:
        device_name = sd.query_devices(device)["name"]
    except Exception:
        device_name = str(device)
    console.print(f"[bold cyan]Listening on device:[/bold cyan] {device_name} (system audio on Windows, mic on macOS)")

    def callback(indata: np.ndarray, frames, time_info, status):
        if status:
            console.print(f"[bold red]Sounddevice status:[/bold red] {status}")
        # indata is (frames, channels) float32
        audio_tensor = torch.from_numpy(indata[:, 0])  # Mono channel 0
        extractor.process_chunk(audio_tensor)

    try:
        with sd.InputStream(
            samplerate=sampling_rate,
            device=device,
            channels=1,
            blocksize=window_size_samples,
            dtype="float32",
            latency="low",
            callback=callback,
        ):
            console.print("[bold green]Streaming started – play system audio (Ctrl+C to stop)[/bold green]")
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping stream...[/bold red]")
        pending_segments = extractor.flush()
        if pending_segments:
            console.print("[bold green]Flushed pending segment(s)[/bold green]")

    console.print(f"\n[bold green]Detected {len(extractor.segments)} speech segments[/bold green]")
    # Optional: save segments similar to existing script if needed

if __name__ == "__main__":
    main()