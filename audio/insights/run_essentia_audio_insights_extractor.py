import os
import shutil
from jet.audio.insights.essentia_audio_insights_extractor import extract_all_insights
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/full_recording.wav"

insights = extract_all_insights(audio_path)

save_file(insights, f"{OUTPUT_DIR}/insights.json")
