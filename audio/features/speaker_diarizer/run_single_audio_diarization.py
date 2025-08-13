import os
import logging
from jet.audio.features.speaker_diarizer import SpeakerDiarizer
from jet.logger import logger


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    """Run speaker diarization on a single audio file."""
    # Define directories
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "mock", "data")
    output_dir = os.path.join(base_dir, "mock", "outputs")
    audio_file = os.path.join(
        base_dir, "mock", "audio", "meeting_recording.wav")

    # Ensure audio file exists
    # Note: Place a WAV file named 'meeting_recording.wav' in the mock/audio folder.
    # The file should be a mono or stereo WAV file, e.g., a 5-minute meeting with 2-4 speakers.
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return

    # Initialize diarizer
    logger.info("Initializing SpeakerDiarizer")
    diarizer = SpeakerDiarizer(data_dir=data_dir, output_dir=output_dir)

    # Perform diarization
    logger.info(f"Processing audio file: {audio_file}")
    annotation = diarizer.diarize(audio_path=audio_file, num_speakers=None)

    # Output results
    logger.info("Diarization completed. Speaker segments:")
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        logger.info(
            f"Speaker {speaker}: {segment.start:.2f}s - {segment.end:.2f}s")


if __name__ == "__main__":
    setup_logging()
    main()
