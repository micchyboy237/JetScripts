# JetScripts/audio/speech/run_extract_speech_speakers_with_transcriptions.py
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Union

import torch
import torchaudio
from faster_whisper import WhisperModel
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from jet.audio.transcribers.utils import segments_to_srt
from jet.audio.utils import AudioInput, resolve_audio_paths
from jet.file.utils import save_file
from jet.logger import logger
from jet.audio.speech.pyannote.speech_speakers_extractor import (
    SpeechSpeakerSegment,
    extract_speech_speakers,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def run_extract_speech_speakers_with_transcriptions(
    audio_inputs: AudioInput,
    *,
    whisper_model: str = "large-v3",
    device: str = "auto",
    compute_type: str = "float16" if torch.cuda.is_available() else "int8",
    language: str | None = None,
    task: str = "transcribe",
    output_dir: Union[str, Path] = OUTPUT_DIR,
    export_segment_audio: bool = True,
    vad_filter: bool = True,
) -> List[Path]:
    """
    Full pipeline:
      • Speaker diarization (pyannote)
      • High-quality transcription per speaker turn (faster-whisper)
      • Per-segment files:
          – segment_XXXXms-YYYYms_SPEAKER_XX.wav
          – segment_XXXXms-YYYYms_SPEAKER_XX.srt
          – segment_XXXXms-YYYYms_SPEAKER_XX.txt  (clean text)
          – translation.txt (if task == "translate")
      • Global files: segments.json, subtitles.srt, info.json

    Returns list of per-file output directories.
    """
    audio_paths = resolve_audio_paths(audio_inputs, recursive=False)
    base_output = Path(output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Whisper model '{whisper_model}' on {device} ({compute_type})")
    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    logger.success("Whisper model loaded")

    created_dirs: List[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing {task.completed}/{task.total} files..."),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Diarize + Transcribe", total=len(audio_paths))

        for idx, audio_path_str in enumerate(
            tqdm(audio_paths, desc="Processing", unit="file", colour="green"), start=1
        ):
            audio_path = Path(audio_path_str)
            file_output_dir = base_output / f"full_{idx:03d}_{audio_path.stem}"
            file_output_dir.mkdir(parents=True, exist_ok=True)
            segments_dir = file_output_dir / "segments"
            segments_dir.mkdir(exist_ok=True)
            created_dirs.append(file_output_dir)

            logger.info(f"Processing → {audio_path.name}")

            # 1. Diarization
            speaker_segments: List[SpeechSpeakerSegment] = extract_speech_speakers(
                str(audio_path),
                time_resolution=3,
            )

            # 2. Load full waveform once (used for both export & Whisper)
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 3. Transcribe each speaker turn individually
            all_whisper_segments = []
            translation_lines: List[str] = []

            for seg in speaker_segments:
                start_sec = seg["start"]
                end_sec = seg["end"]
                speaker = seg["speaker"]

                # Slice audio for this turn
                start_sample = int(start_sec * sample_rate)
                end_sample  = int(end_sec   * sample_rate)
                segment_wave = waveform[:, start_sample:end_sample]

                # Temporary file for Whisper (faster-whisper needs a path)
                tmp_wav = segments_dir / f"tmp_{speaker}_{idx:06d}.wav"
                torchaudio.save(str(tmp_wav), segment_wave, sample_rate)

                # Whisper transcription on the exact speaker turn
                segments, info = model.transcribe(
                    str(tmp_wav),
                    language=language,
                    task=task,
                    vad_filter=vad_filter,
                    word_timestamps=True,
                )
                seg_list = list(segments)
                tmp_wav.unlink()  # clean up

                if not seg_list:
                    text = ""
                else:
                    # Only one segment expected because we fed a short turn
                    text = seg_list[0].text.strip()
                    all_whisper_segments.extend(seg_list)

                # Per-segment files
                prefix = f"{int(start_sec*1000):06d}ms-{int(end_sec*1000):06d}ms_{speaker}"
                seg_wav_path = segments_dir / f"{prefix}.wav"
                seg_srt_path = segments_dir / f"{prefix}.srt"
                seg_txt_path = segments_dir / f"{prefix}.txt"

                # Export audio
                if export_segment_audio:
                    torchaudio.save(str(seg_wav_path), segment_wave, sample_rate)

                # Export SRT (single entry)
                srt_content = segments_to_srt(seg_list) if seg_list else ""
                save_file(srt_content, seg_srt_path)

                # Export clean text
                save_file(text, seg_txt_path)

                # Collect for global translation.txt if needed
                if task == "translate" and text:
                    translation_lines.append(f"[{speaker} {start_sec:.2f}-{end_sec:.2f}] {text}")

            # 4. Global files for the whole audio file
            full_text = " ".join(s.text.strip() for s in all_whisper_segments if s.text.strip())
            info_dict = {
                "source_file": audio_path.name,
                "processed_at": datetime.now().isoformat(),
                "whisper_model": whisper_model,
                "task": task,
                "language": language or info.language,
                "duration_sec": round(audio_path.stat().st_size / (sample_rate * 2), 2),  # rough
                "total_segments": len(speaker_segments),
                "detected_speakers": sorted({s["speaker"] for s in speaker_segments}),
            }

            save_file(info_dict, file_output_dir / "info.json")
            save_file(speaker_segments, file_output_dir / "diarization_segments.json")
            save_file(all_whisper_segments, file_output_dir / "whisper_segments.json")
            save_file(segments_to_srt(all_whisper_segments), file_output_dir / "subtitles.srt")

            with open(file_output_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(full_text)

            if task == "translate" and translation_lines:
                with open(file_output_dir / "translation.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(translation_lines))

            logger.success(f"Saved → {file_output_dir.name}")
            progress.advance(task_id)

    logger.success(f"All done! Outputs in:\n → {base_output.resolve()}")
    return created_dirs


if __name__ == "__main__":
    example_files = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav",
    ]

    run_extract_speech_speakers_with_transcriptions(
        audio_inputs=example_files,
        whisper_model="large-v3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        task="transcribe",           # or "translate" for Japanese → English
        export_segment_audio=True,
    )