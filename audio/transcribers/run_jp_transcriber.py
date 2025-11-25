import os
import shutil
import argparse
from pathlib import Path

from jet.audio.transcribers.jp_en.jp_transcriber import setup_logger, AudioConfig, TranscriberConfig, JapaneseTranscriber
import sounddevice as sd

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# List devices
print(sd.query_devices())

# Select a device
sd.default.device = 1

def get_input_channels() -> int:
    from jet.logger import logger

    device_info = sd.query_devices(sd.default.device[0], 'input')
    channels = device_info['max_input_channels']
    logger.debug(f"Detected {channels} input channels")
    return channels


CHANNELS = min(2, get_input_channels())

print(f"Channels: {CHANNELS}")

# ============================= MAIN =============================
def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Japanese → English with logging + progress")
    parser.add_argument("--device", type=int, default=None, help="Audio device index listed in sd.query_devices()")
    parser.add_argument("--model", type=str, default="turbo", choices=["tiny", "base", "small", "medium", "large-v3", "turbo"])
    parser.add_argument("--chunk", type=float, default=3.0, help="Chunk duration in seconds")
    parser.add_argument("--output-dir", type=str, default=f"{OUTPUT_DIR}/translations", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Minimal console output")
    parser.add_argument("--no-progress", action="store_true", help="Hide tqdm progress bar")
    args = parser.parse_args()

    output_path = Path(args.output_dir).expanduser().resolve()
    logger = setup_logger(output_path, quiet=args.quiet)

    audio_cfg: AudioConfig = {
        "device": args.device,
        "samplerate": 16000,
        "chunk_duration": args.chunk,
        "channels": CHANNELS,
    }

    # Initial compute_type (overridden in __init__ for Apple Silicon)
    compute_type = "int8" if args.model in ("tiny", "base") else "float16"

    trans_cfg: TranscriberConfig = {
        "model_size": args.model,
        "compute_type": compute_type,
        "language": "ja",
        "task": "translate",
        "device": "auto",  # Placeholder; detected in __init__
    }

    transcriber = JapaneseTranscriber(
        audio_cfg=audio_cfg,
        trans_cfg=trans_cfg,
        output_dir=output_path,
        logger=logger,
        show_progress=not args.no_progress and not args.quiet,
    )
    transcriber.start()


if __name__ == "__main__":
    main()