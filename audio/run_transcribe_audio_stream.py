import subprocess
import sys
import argparse
import threading
import time
import signal
from typing import Optional, List
from jet.audio.e2e.send_mic_stream import send_mic_stream
from jet.audio.transcription_utils import initialize_whisper_model, transcribe_audio_stream
from jet.audio.utils import capture_and_save_audio
from jet.logger import logger


def run_transcription(audio_file: str, language: str = "en"):
    """Run transcription on the audio file in real-time."""
    try:
        model = initialize_whisper_model(
            model_size="small", device="auto", compute_type="float16")
        for start, end, text in transcribe_audio_stream(audio_file, model, language=language):
            print(f"[{start:.2f}s -> {end:.2f}s] {text}")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)


def main(
    receiver_ip: Optional[str] = None,
    port: Optional[int] = None,
    sample_rate: int = 44100,
    channels: int = 2,
    file_prefix: str = "recording",
    device_index: str = "1",
    language: str = "en"
):
    """Orchestrate audio capture and real-time transcription."""
    # Generate the audio file name based on the file_prefix
    from jet.audio.utils import get_next_file_suffix
    start_suffix = get_next_file_suffix(file_prefix)
    audio_file = f"{file_prefix}_{start_suffix:05d}.wav"

    # Store processes for cleanup
    processes: List[subprocess.Popen] = []

    def signal_handler(sig, frame):
        logger.info("Terminating FFmpeg processes and exiting...")
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"FFmpeg process {process.pid} did not terminate gracefully, killing...")
                process.kill()
        sys.exit(0)

    # Set signal handler in the main thread
    signal.signal(signal.SIGINT, signal_handler)

    # Start audio capture in a separate thread
    audio_thread = threading.Thread(
        target=lambda: processes.append(
            capture_and_save_audio(sample_rate, channels,
                                   file_prefix, device_index)
        ),
        daemon=True
    )
    audio_thread.start()

    # Wait briefly to ensure the audio file is created
    time.sleep(0.5)

    # Start transcription in the main thread
    run_transcription(audio_file, language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run microphone audio capture and real-time transcription."
    )
    parser.add_argument("--receiver-ip", type=str, default=None,
                        help="IP address of the RTP receiver (optional)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port number for RTP streaming (optional)")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="Audio sample rate (Hz, default: 44100)")
    parser.add_argument("--channels", type=int, default=2, choices=[1, 2],
                        help="Number of audio channels (1=mono, 2=stereo, default: 2)")
    parser.add_argument("--file-prefix", type=str, default="recording",
                        help="Prefix for output WAV file (default: recording)")
    parser.add_argument("--device-index", type=str, default="1",
                        help="avfoundation device index for microphone (default: 1)")
    parser.add_argument("--language", type=str, default="en",
                        help="Language for transcription (default: en)")
    args = parser.parse_args()

    if (args.receiver_ip is None) != (args.port is None):
        parser.error(
            "Both --receiver-ip and --port must be provided together or both omitted")

    main(
        args.receiver_ip,
        args.port,
        args.sample_rate,
        args.channels,
        args.file_prefix,
        args.device_index,
        args.language
    )
