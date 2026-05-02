import os
import shutil
import torchaudio

from silero_vad.utils_vad import read_audio

from jet.audio.speech.silero.speech_utils import check_speech_waves
from jet.audio.utils import resolve_audio_paths
from jet.file.utils import save_file
from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

audio_inputs = [
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/speech/generated/run_extract_speech_timestamps/segments",
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/results/full_recording.wav",
]
audio_paths = resolve_audio_paths(audio_inputs, recursive=True)

segments_dir = f"{OUTPUT_DIR}/segments"

for idx, audio_file in enumerate(audio_paths):
    waveform = read_audio(audio_file, sampling_rate=16000).unsqueeze(0)
    speech_ts_and_probs = extract_speech_timestamps(audio_file, with_scores=True)
    if isinstance(speech_ts_and_probs, tuple):
        speech_ts, speech_probs = speech_ts_and_probs
    else:
        speech_ts, speech_probs = speech_ts_and_probs, []
    threshold = 0.3
    speech_waves = check_speech_waves(speech_probs, threshold=threshold)

    out_dir = f"{segments_dir}/segment_{int(idx) + 1:03d}"
    waves_dir = os.path.join(out_dir, "waves")
    os.makedirs(waves_dir, exist_ok=True)

    save_file(speech_waves, f"{out_dir}/speech_waves.json")
    save_file(speech_probs, f"{out_dir}/speech_probs.json")
    save_file(speech_ts, f"{out_dir}/speech_ts.json")

    # Save each speech wave as a separate WAV file regardless of metadata validity
    for wave in speech_waves:
        frame_start = wave["details"]["frame_start"]
        frame_end = wave["details"]["frame_end"]
        start_sample = frame_start * 512
        end_sample = (frame_end + 1) * 512
        wave_audio = waveform[:, start_sample:end_sample]
        wav_path = os.path.join(waves_dir, f"sound_{frame_start}_{frame_end}.wav")
        torchaudio.save(wav_path, wave_audio, sample_rate=16000)
