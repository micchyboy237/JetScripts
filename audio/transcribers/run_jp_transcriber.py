import os
import shutil
import argparse
from pathlib import Path

from jet.audio.transcribers.jp_en.jp_transcriber import setup_logger, AudioConfig, TranscriberConfig, JapaneseTranscriber
from jet.audio.utils import get_input_channels
import sounddevice as sd

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# List devices
print(sd.query_devices())

# Select a device
sd.default.device = 1

CHANNELS = min(2, get_input_channels())

print(f"Channels: {CHANNELS}")

# ============================= MAIN =============================
def main() -> None:
    parser = argparse.ArgumentParser(description="Japanese → English Real-time • Zero Loss • Max Quality")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--chunk", type=float, default=2.0, help="Audio chunk size in seconds (1.5–3.0 recommended)")
    parser.add_argument("--output-dir", type=str, default=f"{OUTPUT_DIR}/translations")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_dir).expanduser().resolve()
    logger = setup_logger(output_path, args.quiet)

    audio_cfg: AudioConfig = {
        "device": args.device,
        "samplerate": 16000,
        "chunk_duration": args.chunk,
        "channels": CHANNELS,
    }

    trans_cfg: TranscriberConfig = {
        "model_size": "large-v3",
        "compute_type": "int16",
        "language": "ja",
        "task": "translate",   # ← This guarantees English output
        "device": "auto",
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