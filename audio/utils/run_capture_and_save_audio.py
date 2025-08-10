import argparse
import signal
import sys
import time
from jet.audio.utils import capture_and_save_audio
from jet.logger import logger


def main():
    """Capture audio from a microphone and save as segmented WAV files."""
    parser = argparse.ArgumentParser(
        description="Record audio from a microphone and save as segmented WAV files."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate in Hz (default: 44100)"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of audio channels (1=mono, 2=stereo, default: 2)"
    )
    parser.add_argument(
        "--segment-time",
        type=int,
        default=30,
        help="Duration of each WAV segment in seconds (default: 30)"
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default="recording",
        help="Prefix for output WAV files (default: recording)"
    )
    parser.add_argument(
        "--device-index",
        type=str,
        default="1",
        help="AVFoundation device index for microphone (default: 1)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum duration of audio to save a segment (seconds, default: 1.0)"
    )
    parser.add_argument(
        "--segment-flush-interval",
        type=int,
        default=5,
        help="Interval to flush audio to new segment in seconds (default: 5)"
    )
    args = parser.parse_args()

    logger.info(
        f"Starting audio capture with device index {args.device_index}...")
    try:
        process = capture_and_save_audio(
            sample_rate=args.sample_rate,
            channels=args.channels,
            segment_time=args.segment_time,
            file_prefix=args.file_prefix,
            device_index=args.device_index,
            min_duration=args.min_duration,
            segment_flush_interval=args.segment_flush_interval
        )

        def signal_handler(sig, frame):
            logger.info("Stopping audio capture...")
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"FFmpeg process {process.pid} did not terminate gracefully, killing...")
                process.kill()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        logger.info(
            f"Recording audio to {args.file_prefix}_*.wav. Press Ctrl+C to stop.")
        while True:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(
                    f"FFmpeg process {process.pid} exited unexpectedly: {stderr}")
                sys.exit(1)
            time.sleep(0.1)

    except Exception as e:
        logger.error(f"Error during audio capture: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
