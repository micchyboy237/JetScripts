import argparse
from jet.audio.utils import get_next_file_suffix
from jet.logger import logger


def main():
    """Determine the next available file suffix for audio recordings."""
    parser = argparse.ArgumentParser(
        description="Get the next available file suffix for a given prefix."
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default="recording",
        help="Prefix for audio files (default: recording)"
    )
    args = parser.parse_args()

    logger.info(f"Finding next file suffix for prefix '{args.file_prefix}'...")
    try:
        suffix = get_next_file_suffix(args.file_prefix)
        logger.info(f"Next available suffix: {suffix}")
        print(f"Next file will be named: {args.file_prefix}_{suffix:05d}.wav")
    except Exception as e:
        logger.error(f"Error finding next suffix: {e}")
        exit(1)


if __name__ == "__main__":
    main()
