import shutil
from pathlib import Path

from jet.audio.utils import combine_audio_files

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_paths = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/sub_audio/mid_5-10s_recording_1_speaker.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/sub_audio/mid_10-15s_recording_3_speakers.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/sub_audio/start_5s_recording_1_speaker.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/sub_audio/start_5s_recording_3_speakers.wav",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/sub_audio/start_15s_recording_1_speaker.wav",
]

output_path = OUTPUT_DIR / "combined_audio.wav"

input_paths = [Path(p) for p in input_paths]

combine_audio_files(input_paths, output_path)
