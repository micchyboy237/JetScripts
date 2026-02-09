import shutil
from pathlib import Path

import soundfile as sf
from jet.audio.norm import normalize_speech_loudness
from jet.audio.utils import extract_audio_segment

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/last_5_mins.wav"

# Audio duration: 5 min 20 sec = 320 sec
TOTAL_DURATION = 320.0  # seconds

# Last 10 seconds
start = TOTAL_DURATION - 10.0  # 310.0
end = TOTAL_DURATION  # 320.0

# Extract from raw input audio
segment, sr = extract_audio_segment(INPUT_AUDIO, start=start, end=end)
output_path = OUTPUT_DIR / f"recording_missav_{end - start}s.wav"
sf.write(output_path, segment, sr)
print("Extracted from raw input. Saved at:")
print(output_path)

# Extract from normalized audio
norm_segment = normalize_speech_loudness(segment, sr)
norm_output_path = OUTPUT_DIR / f"recording_missav_{end - start}s_norm.wav"
sf.write(norm_output_path, norm_segment, sr)
print("Extracted from normalized input. Saved at:")
print(norm_output_path)
