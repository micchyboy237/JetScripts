import shutil
from pathlib import Path

import librosa
import numpy as np
from jet.audio.speech.speechbrain.vad import SpeechBrainVAD
from jet.file.utils import save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def chunk_audio(
    audio: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """
    Convert 1D audio into (num_chunks, chunk_size).
    Pads last chunk if necessary.
    """
    total_samples = len(audio)
    remainder = total_samples % chunk_size

    if remainder != 0:
        pad_width = chunk_size - remainder
        audio = np.pad(audio, (0, pad_width))

    return audio.reshape(-1, chunk_size)


SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 512

audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"

audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

chunk_size = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000))

chunks = chunk_audio(audio, chunk_size=chunk_size)

vad = SpeechBrainVAD()
all_speech_probs = vad.get_speech_probs(chunks)

save_file(all_speech_probs, OUTPUT_DIR / "all_speech_probs.json")
