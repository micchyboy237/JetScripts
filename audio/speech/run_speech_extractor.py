from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from jet.audio.speech.silero.speech_extractor import (
    extract_frame_energy,
    process_audio,
    save_energy_plot,
    save_per_segment_data,
    save_probs_plot,
)
from jet.file.utils import save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    audio_file = Path(
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/utils/generated/run_extract_audio_segment/recording_missav_10.0s_norm.wav"
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/utils/generated/run_extract_audio_segment/recording_missav_10.0s.wav"
    )

    results = process_audio(
        audio=audio_file,
        sampling_rate=16000,
        threshold=0.3,
        low_threshold=0.1,
    )

    results, full_waveform = results  # unpack

    # Save results
    save_file(results["probs"], OUTPUT_DIR / "probs.json")
    save_file(results["segments"], OUTPUT_DIR / "segments.json")
    save_file(results["meta"], OUTPUT_DIR / "meta.json")

    # Generate and save plot
    save_probs_plot(
        probs=np.array(results["probs"]),
        sampling_rate=16000,
        segments=results["segments"],
        output_path=OUTPUT_DIR / "probs_plot.png",
    )

    energy = extract_frame_energy(audio_file, sampling_rate=16000)
    save_file(energy, OUTPUT_DIR / "energy.json")

    # Generate and save energy plot
    save_energy_plot(
        energies=energy,
        sampling_rate=16000,
        segments=results["segments"],
        output_path=OUTPUT_DIR / "energy_plot.png",
    )

    # Reconstruct SpeechSegment objects for per-segment saving
    segments_objs = results["segments"]

    if segments_objs:
        save_per_segment_data(
            waveform=full_waveform,
            probs=np.array(results["probs"]),
            energies=energy,
            segments=segments_objs,
            meta=results["meta"],
            sampling_rate=16000,
            output_dir=OUTPUT_DIR / "segments",
        )
