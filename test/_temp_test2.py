# download the pipeline from Huggingface
from pathlib import Path
import shutil
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", 
    token="{huggingface-token}")

# run the pipeline locally on your computer
audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251212_041845.wav"
output = pipeline(audio_file)

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")

output_dir = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Diarization output written under: {output_dir.resolve()}")