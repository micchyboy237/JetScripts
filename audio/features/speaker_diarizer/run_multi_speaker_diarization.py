import os
from pyannote.core import Annotation
from omegaconf import OmegaConf
from jet.audio.features.speaker_diarizer import SpeakerDiarizer
from jet.logger import logger


def main():
    """
    Run speaker diarization with a known number of speakers and custom configuration.
    Example use case: Diarizing a meeting with 4 speakers using a custom config.

    Audio file requirements:
    - Place a WAV file (e.g., mono/stereo, 16kHz) in the mock folder.
    - Example: A 5-10 minute meeting recording with 4 distinct speakers.
    - Path: /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/features/speaker_diarizer/mock/meeting_sample.wav

    Custom config requirements:
    - Adjusts VAD and clustering parameters for better multi-speaker detection.
    - Saved as custom_config.yaml in the data directory.
    """
    # Define directories
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/features/speaker_diarizer"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")
    audio_path = os.path.join(base_dir, "mock", "meeting_sample.wav")
    # config_path = os.path.join(data_dir, "custom_config.yaml")
    config_path = None

    # # Create custom config
    # custom_config = {
    #     "diarizer": {
    #         "vad": {
    #             "parameters": {
    #                 "onset": 0.7,  # Tighter onset for clearer speech detection
    #                 "offset": 0.5,
    #                 "pad_offset": -0.1
    #             }
    #         },
    #         "clustering": {
    #             "parameters": {
    #                 "oracle_num_speakers": True,  # Use known number of speakers
    #                 "max_num_speakers": 4
    #             }
    #         }
    #     }
    # }
    # os.makedirs(data_dir, exist_ok=True)
    # OmegaConf.save(custom_config, config_path)
    # logger.info(f"Custom config saved to {config_path}")

    # Initialize diarizer with custom config
    diarizer = SpeakerDiarizer(
        data_dir=data_dir, output_dir=output_dir, config_path=config_path)
    logger.info(f"Processing audio file: {audio_path} with 4 speakers")

    # Perform diarization
    try:
        annotation: Annotation = diarizer.diarize(audio_path, num_speakers=4)
        logger.info(f"Diarization completed. Output saved in {output_dir}")

        # Print results
        print("Diarization Results (4 speakers):")
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            start = segment.start
            end = segment.end
            print(f"Speaker {speaker}: {start:.2f}s - {end:.2f}s")
    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
