


import json
import time
from pathlib import Path
from rich import print as rprint

from jet.audio.transcribers.base_client import transcribe_audio

if __name__ == "__main__":
    file_path = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps/segment_002/sound.wav")
    # file_path = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/segments/segment_002/sound.wav")

    start1 = time.perf_counter()
    result1 = transcribe_audio(file_path)
    end1 = time.perf_counter()
    rprint("[bold green]Multipart result:[/bold green]")
    rprint(json.dumps(result1, indent=2, ensure_ascii=False))
    print(f"[bold green]upload_file_multipart duration:[/bold green] {end1 - start1:.3f} seconds")
