import argparse
from jet.audio.utils import list_avfoundation_devices
from jet.logger import logger


def main():
    """List available AVFoundation devices for audio input."""
    parser = argparse.ArgumentParser(
        description="List available microphones using AVFoundation."
    )
    args = parser.parse_args()

    logger.info("Listing available AVFoundation devices...")
    try:
        devices = list_avfoundation_devices()
        logger.info("Available devices:")
        print(devices)
    except SystemExit:
        logger.error(
            "Failed to list devices. Check FFmpeg installation and microphone permissions.")
        exit(1)


if __name__ == "__main__":
    main()
