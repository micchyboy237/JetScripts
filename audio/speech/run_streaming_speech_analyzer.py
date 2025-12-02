import os
import shutil

from jet.audio.speech.silero.silero_vad_stream import SileroVADStreamer

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    streamer = SileroVADStreamer(
        output_dir=OUTPUT_DIR,   # ← enables saving
        save_segments=True,
    )
    streamer.start()