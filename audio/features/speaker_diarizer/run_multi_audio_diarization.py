import os
import logging
from typing import List
from jet.audio.features.speaker_diarizer import SpeakerDiarizer
from jet.logger import logger


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_audio_files(audio_dir: str) -> List[str]:
    """Retrieve all WAV files from the audio directory."""
    # Note: Place WAV files in the mock/audio folder, e.g., 'podcast_ep1.wav', 'podcast_ep2.wav'.
    # Each file should be a mono or stereo WAV file, e.g., podcast episodes with 2 speakers.
    return [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]


def main():
    """Run speaker diarization on multiple audio files."""
    # Define directories
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "mock", "data")
    output_dir = os.path.join(base_dir, "mock", "outputs")
    audio_dir = os.path.join(base_dir, "mock", "audio")

    # Initialize diarizer
    logger.info("Initializing SpeakerDiarizer")
    diarizer = SpeakerDiarizer(data_dir=data_dir, output_dir=output_dir)

    # Get audio files
    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        logger.error(f"No WAV files found in {audio_dir}")
        return

    # Process each audio file
    num_speakers = 2  # Example: Assume 2 speakers for podcasts
    for audio_file in audio_files:
        logger.info(f"Processing audio file: {audio_file}")
        annotation = diarizer.diarize(
            audio_path=audio_file, num_speakers=num_speakers)

        # Output results
        logger.info(
            f"Diarization completed for {audio_file}. Speaker segments:")
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            logger.info(
                f"Speaker {speaker}: {segment.start:.2f}s - {segment.end:.2f}s")


if __name__ == "__main__":
    setup_logging()
    main()
