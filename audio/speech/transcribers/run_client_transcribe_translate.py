


from pathlib import Path

from jet.audio.transcribers.base_client import transcribe_audio


async def main():
    file_path = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/data/sound.wav")

    print("[bold magenta]Multipart upload (file path)[/bold magenta]")
    await transcribe_audio(file_path)

    print("\n" + "─" * 50 + "\n")

    print("[bold cyan]Raw bytes upload (in-memory)[/bold cyan]")
    raw_bytes = file_path.read_bytes()
    await transcribe_audio(raw_bytes, filename="sound_from_memory.wav")

    # Example with numpy array (e.g. from librosa, torchaudio, etc.)
    # import librosa
    # y, sr = librosa.load(file_path, sr=16000)
    # await transcribe_audio(y, filename="numpy_array.wav")


if __name__ == "__main__":
    asyncio.run(main())