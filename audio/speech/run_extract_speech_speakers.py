# JetScripts/audio/speech/run_extract_speech_speakers.py
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Union

import torchaudio
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from jet.audio.utils import AudioInput, resolve_audio_paths
from jet.file.utils import save_file
from jet.logger import logger
from jet.audio.speech.pyannote.speech_speakers_extractor import (
    SpeechSpeakerSegment,
    extract_speech_speakers,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def run_extract_speech_speakers(
    audio_inputs: AudioInput,
    *,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    export_segment_audio: bool = True,
    time_resolution: int = 2,
    recursive: bool = False,
) -> List[Path]:
    """
    Run speaker diarization on one or many audio files and save results + optional per-segment WAVs.

    Args:
        audio_inputs: Single file, list of files or directory (non-recursive)
        output_dir: Base directory where per-file folders will be created
        export_segment_audio: If True saves each speaker turn as a separate .wav file
        time_resolution: Decimal places for start/end timestamps

    Returns:
        List of created per-file output directories
    """
    audio_paths = resolve_audio_paths(audio_inputs, recursive=recursive)
    base_output = Path(output_dir)
    shutil.rmtree(base_output, ignore_errors=True)
    base_output.mkdir(parents=True, exist_ok=True)

    logger.info("Starting speaker diarization on %d file(s)", len(audio_paths))

    created_dirs: List[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing {task.completed}/{task.total} files..."),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Diarization", total=len(audio_paths))

        for idx, audio_path_str in enumerate(tqdm(audio_paths, desc="Diarizing", unit="file", colour="magenta"), start=1):
            audio_path = Path(audio_path_str)
            file_output_dir = base_output / f"diarized_{idx:03d}_{audio_path.stem}"
            file_output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(file_output_dir)

            logger.info(f"Processing → {audio_path.name}")

            # 1. Run diarization
            segments: List[SpeechSpeakerSegment] = extract_speech_speakers(
                audio=str(audio_path),
                time_resolution=time_resolution,
            )

            # 2. Save metadata
            info = {
                "source_file": audio_path.name,
                "processed_at": datetime.now().isoformat(),
                "total_segments": len(segments),
                "speakers": sorted({seg["speaker"] for seg in segments}),
            }
            save_file(info, file_output_dir / "info.json")
            save_file(segments, file_output_dir / "segments.json")

            # 3. Optional per-segment audio export
            if export_segment_audio and segments:
                waveform, sample_rate = torchaudio.load(audio_path)

                # Ensure mono (downmix if multi-channel)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)  # → (1, samples)

                for seg in segments:
                    start_sample = int(seg["start"] * sample_rate)
                    end_sample = int(seg["end"] * sample_rate)

                    # Slice the waveform for this speaker turn
                    segment_wave = waveform[:, start_sample:end_sample]  # (1, segment_samples)

                    segment_name = (
                        f"{int(seg['start']*1000):06d}ms-"
                        f"{int(seg['end']*1000):06d}ms_"
                        f"{seg['speaker']}.wav"
                    )
                    out_path = file_output_dir / "segments" / segment_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    torchaudio.save(
                        str(out_path),
                        segment_wave,           # already has channel dim
                        sample_rate,
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )

                logger.success(f"Exported {len(segments)} segment WAVs")

            logger.success(f"Saved → {file_output_dir.name}")
            progress.advance(task_id)

    logger.success(f"All done! Outputs in:\n → {base_output.resolve()}")
    return created_dirs


if __name__ == "__main__":
    example_files = [
        # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav",
    ]

    run_extract_speech_speakers(
        audio_inputs=example_files,
        export_segment_audio=True,
        recursive=True,
    )