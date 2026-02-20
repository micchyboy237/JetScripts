import shutil
from pathlib import Path

import soundfile as sf
from jet.audio.audio_duration import get_audio_duration
from jet.audio.norm import normalize_speech_loudness
from jet.audio.utils import extract_audio_segment

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_per_speech/last_5_mins.wav"

start = 120
end = None

if end is None:
    end = get_audio_duration(INPUT_AUDIO)

# Extract from raw input audio
segment, sr = extract_audio_segment(INPUT_AUDIO, start=start, end=end)
output_path = OUTPUT_DIR / "recording_missav.wav"
sf.write(output_path, segment, sr)
print("Extracted from raw input. Saved at:")
print(output_path)

# Extract from normalized audio
norm_segment = normalize_speech_loudness(segment, sr)
norm_output_path = OUTPUT_DIR / "recording_missav_norm.wav"
sf.write(norm_output_path, norm_segment, sr)
print("Extracted from normalized input. Saved at:")
print(norm_output_path)
