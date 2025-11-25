import asyncio
import platform
from jet.audio.audio_system_transcriber import AudioSystemTranscriber

async def main():
    """
    async def main():
        # Instantiates AudioSystemTranscriber with default settings
        # Infinite loop that awaits a new transcription on every speech segment
        # Prints the result or "No speech detected."
        # Catches KeyboardInterrupt for clean shutdown
    """
    transcriber = AudioSystemTranscriber()
    while True:
        try:
            transcription = await transcriber.capture_and_transcribe()
            if transcription:
                print(f"Transcription: {transcription}")
            else:
                print("No speech detected.")
            await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping transcription.")
            break

"""
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
# Platform-specific asyncio entry point (standard Python vs Pyodide/Emscripten)
"""
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())