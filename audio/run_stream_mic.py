import os
from datetime import datetime
import numpy as np
from jet.audio.record_mic import save_wav_file
from jet.audio.stream_mic import stream_non_silent_audio

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)


def main():
    """
    Stream non-silent audio from microphone and save to a WAV file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    # Collect non-silent chunks
    audio_chunks = []
    for chunk in stream_non_silent_audio(
        silence_threshold=None,  # Auto-calibrate
        chunk_duration=0.5,
        silence_duration=2.0
    ):
        audio_chunks.append(chunk)

    if not audio_chunks:
        print("No non-silent audio captured.")
        return

    # Concatenate chunks and save
    audio_data = np.concatenate(audio_chunks, axis=0)
    save_wav_file(output_file, audio_data)
    print(f"Streamed audio saved to {output_file}")


if __name__ == "__main__":
    main()
