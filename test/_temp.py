import json
import torch

# --------------------------------------------------
# Load Silero VAD JIT model + utilities
# --------------------------------------------------
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)

(
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks,
) = utils

# --------------------------------------------------
# Read audio
# --------------------------------------------------
audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
wav = read_audio(audio_file, sampling_rate=16000)
speech_ts = get_speech_timestamps(wav, model, return_seconds=True, time_resolution=4)

# Add duration to each segment
for seg in speech_ts:
    seg["start"] = round(seg["start"], 3)
    seg["end"] = round(seg["end"], 3)
    seg["duration"] = round(seg["end"] - seg["start"], 3)

print(json.dumps(speech_ts, indent=2))
