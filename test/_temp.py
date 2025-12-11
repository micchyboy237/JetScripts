import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Community-1 open-source speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="HUGGINGFACE_ACCESS_TOKEN")

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline (with optional progress hook)
audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251212_041845.wav"
with ProgressHook() as hook:
    output = pipeline(audio_file, hook=hook)  # runs locally

# print the result
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
