import os
import shutil
from jet.audio.large_audio_transcriber import AudioTranscriber
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

DEFAULT_AUDIO_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/video/generated/run_youtube_scrape_info/sen__legarda_sa_kwento_ni_bamban_mayor_guo__paulit_ulit_parang_memoryado/audio.mp3"

if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Transcribe a large audio file with overlap.")
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=DEFAULT_AUDIO_FILE,
        type=str,
        help=f"Path to the audio file to transcribe (default: {DEFAULT_AUDIO_FILE})"
    )
    args = parser.parse_args()

    # Path to the downloaded audio file
    audio_file = args.audio_file
    audio_file = Path(audio_file)
    model_size = "small"  # Using small model
    device = "cpu"  # Default to CPU for compatibility with Mac M1
    segment_duration = 5.0  # 5-second segments
    overlap_duration = 1.0  # 1.0-second overlap to prevent information loss
    language = "en"  # Specify language (optional, set to English)

    try:
        # Initialize the transcriber
        transcriber = AudioTranscriber(
            model_size=model_size,
            device=device,
            compute_type="int8"  # Optimized for CPU
        )

        # Verify the audio file exists
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file '{audio_file}' not found.")

        # Transcribe the audio file
        print(f"Transcribing {audio_file} with {overlap_duration}s overlap...")
        segments = []
        for text, start_time, end_time in transcriber.transcribe_audio(
            audio_path=audio_file,
            segment_duration=segment_duration,
            overlap_duration=overlap_duration,
            language=language
        ):
            # Format and print the transcription with timestamps
            print(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}")
            segments.append({
                "text": text,
                "start_time": start_time,
                "end_time": end_time
            })
            save_file(segments, f"{OUTPUT_DIR}/segments.json")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        sys.exit(1)
