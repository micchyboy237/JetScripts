import asyncio
import sys
import argparse

from jet.audio.audio_file_transcriber import AudioFileTranscriber

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using AudioFileTranscriber.")
    parser.add_argument("audio_file", type=str,
                        help="Path to the audio file to transcribe (e.g., WAV, MP3)")
    args = parser.parse_args()

    transcriber = AudioFileTranscriber(model_size="small")
    print(f"Transcribing file: {args.audio_file}")
    try:
        result = asyncio.get_event_loop().run_until_complete(
            transcriber.transcribe_from_file(args.audio_file)
        )
        if result:
            print("Transcription:", result)
        else:
            print("No audio captured or no transcription result.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)
