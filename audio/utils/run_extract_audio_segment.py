from pathlib import Path
import shutil
import soundfile as sf
from jet.audio.utils import extract_audio_segment

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_missav_5mins.wav"
NORM_INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/preprocessors/recording_missav_5mins_norm.wav"

start = 0.0
end = 60.0

# Extract first minute from raw input audio
segment, sr = extract_audio_segment(INPUT_AUDIO, start=start, end=end)
output_path = OUTPUT_DIR / "recording_missav_1min.wav"
sf.write(output_path, segment, sr)
print("Extracted first minute of raw input. Saved at:")
print(output_path)

# Extract first minute from normalized audio
norm_segment, norm_sr = extract_audio_segment(NORM_INPUT_AUDIO, start=start, end=end)
norm_output_path = OUTPUT_DIR / "recording_missav_1min_norm.wav"
sf.write(norm_output_path, norm_segment, norm_sr)
print("Extracted first minute of normalized input. Saved at:")
print(norm_output_path)
