import os
from pyannote.core import Annotation, Segment
from jet.audio.features.speaker_diarizer import SpeakerDiarizer
from jet.logger import logger


def main():
    """
    Run speaker diarization on a single audio file with unknown number of speakers.
    Example use case: Diarizing a podcast or interview recording.

    Audio file requirements:
    - Place a WAV file (e.g., mono/stereo, 16kHz) in the mock folder.
    - Example: A 2-5 minute podcast with 2-3 speakers.
    - Path: /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/features/speaker_diarizer/mock/podcast_sample.wav
    """
    # Define directories
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/features/speaker_diarizer"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")
    audio_path = os.path.join(base_dir, "mock", "podcast_sample.wav")

    # Initialize diarizer
    diarizer = SpeakerDiarizer(data_dir=data_dir, output_dir=output_dir)
    logger.info(f"Processing audio file: {audio_path}")

    # Perform diarization
    try:
        annotation: Annotation = diarizer.diarize(audio_path)
        logger.info(f"Diarization completed. Output saved in {output_dir}")

        # Print results
        print("Diarization Results:")
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            start = segment.start
            end = segment.end
            print(f"Speaker {speaker}: {start:.2f}s - {end:.2f}s")
    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
