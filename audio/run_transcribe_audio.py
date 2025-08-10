import asyncio
from jet.audio.system.transcribe_system_audio import AudioTranscriber

if __name__ == "__main__":
    # Example usage: transcribe audio from system using the "small" model
    transcriber = AudioTranscriber(
        model_size="small", sample_rate=16000, chunk_duration=1.0)
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
