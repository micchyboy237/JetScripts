from jet.audio.helpers.energy import compute_energies
from jet.audio.helpers.silence import calibrate_silence_threshold
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_2_speakers.wav"

silence_threshold = calibrate_silence_threshold()
energies = compute_energies(audio_file, silence_threshold=silence_threshold)

for e in energies[:10]:
    print(e)
# → {'start_s': 0.0, 'end_s': 0.5, 'energy': 0.012345, 'is_silent': True}
save_file({
    "energies_count": len(energies),
    "silence_threshold": silence_threshold,
}, f"{OUTPUT_DIR}/info.json")
save_file(energies, f"{OUTPUT_DIR}/energies.json")