import os
import shutil

from jet.audio.speech.silero.speech_analyzer import SileroVADAnalyzer
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    analyzer = SileroVADAnalyzer(
        threshold=0.5,
        speech_pad_ms=30,
    )
    probs, segments = analyzer.analyze(audio_file)
    analyzer.plot_insights(probs, segments, audio_file, OUTPUT_DIR)
    analyzer.save_json(segments, OUTPUT_DIR, audio_file)

    save_file(probs, f"{OUTPUT_DIR}/probs.json")
    save_file(segments, f"{OUTPUT_DIR}/segments.json")