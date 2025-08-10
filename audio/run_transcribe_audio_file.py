import asyncio
import os
import shutil
import sys
import argparse

from jet.audio.audio_file_transcriber import AudioFileTranscriber

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
# shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using AudioFileTranscriber."
    )
    parser.add_argument("audio_file", type=str,
                        help="Path to the audio file to transcribe (e.g., WAV, MP3)")
    parser.add_argument("output_dir", type=str, nargs='?', default=OUTPUT_DIR,
                        help="Optional directory to save transcription text file")
    args = parser.parse_args()
    transcriber = AudioFileTranscriber(
        model_size="small", sample_rate=None)  # Explicitly set to None
    print(f"Transcribing file: {args.audio_file}")
    try:
        result = asyncio.run(  # Changed from get_event_loop().run_until_complete
            transcriber.transcribe_from_file(
                args.audio_file, output_dir=args.output_dir)
        )
        if result:
            print("Transcription:", result)
        else:
 sequencing
            print("No audio captured edged or no transcription result.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)