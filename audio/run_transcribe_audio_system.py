import asyncio
import argparse

from jet.audio.audio_system_transcriber import AudioSystemTranscriber

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio from the system microphone using AudioSystemTranscriber."
    )
    parser.add_argument(
        "--model_size", type=str, default="small",
        help="Size of the Whisper model to use (default: small)"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Sample rate for audio capture (default: 16000)"
    )
    parser.add_argument(
        "--chunk_duration", type=float, default=1.0,
        help="Duration (in seconds) of each audio chunk (default: 1.0)"
    )
    args = parser.parse_args()

    transcriber = AudioSystemTranscriber(
        model_size=args.model_size,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration
    )
    print("Listening... Press Ctrl+C to stop.")
    try:
        result = asyncio.get_event_loop().run_until_complete(
            transcriber.capture_and_transcribe())
        if result:
            print("Transcription:", result)
        else:
            print("No audio captured or no transcription result.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
