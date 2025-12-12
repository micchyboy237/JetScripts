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
audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers.wav"
wav = read_audio(audio_file, sampling_rate=16000)

# Streaming window size expected by Silero (512 samples @ 16 kHz)
CHUNK = 512

# --------------------------------------------------
# Create iterator for streaming VAD
# --------------------------------------------------
vad_iterator = VADIterator(
    model=model,
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=100,
    speech_pad_ms=30,
)

timestamps = []
current_segment = {}

# --------------------------------------------------
# Simulate streaming
# --------------------------------------------------
for i in range(0, len(wav), CHUNK):
    chunk = wav[i:i+CHUNK]

    # Pad last chunk if smaller than 512 samples
    if len(chunk) < CHUNK:
        chunk = torch.nn.functional.pad(chunk, (0, CHUNK - len(chunk)))

    # Feed chunk into the iterator
    out = vad_iterator(chunk, return_seconds=True)

    if out is None:
        continue

    # Start-of-speech event
    if "start" in out:
        current_segment = {"start": out["start"]}

    # End-of-speech event
    if "end" in out:
        current_segment["end"] = out["end"]
        timestamps.append(current_segment)
        current_segment = {}

# --------------------------------------------------
# Print detected segments
# --------------------------------------------------
print("Detected speech:")
for ts in timestamps:
    print(ts)
