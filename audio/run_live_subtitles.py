# JetScripts/audio/run_record_mic.py
import json
from pathlib import Path
import shutil
import sys
from threading import Thread
from PyQt6.QtWidgets import QApplication
import numpy as np

from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.record_mic_speech_detection import record_from_mic
from jet.audio.speech.overlay_utils import append_to_combined_srt, write_srt_file
from jet.audio.speech.silero.speech_types import SpeechSegment
from jet.audio.speech.wav_utils import get_wav_bytes, save_wav_file
from jet.audio.transcribers.transcription_pipeline import TranscriptionPipeline
from jet.file.utils import save_file
from jet.logger import logger
from jet.overlays.subtitle_overlay import SubtitleOverlay

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_segment_data(speech_seg: SpeechSegment, seg_audio_np: np.ndarray):
    segment_root = Path(OUTPUT_DIR) / "segments"
    segment_root.mkdir(parents=True, exist_ok=True)

    # Use the canonical idx from the speech detector as segment number
    seg_number: int = speech_seg["idx"] + 1
    seg_dir = segment_root / f"segment_{seg_number:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    wav_path = seg_dir / "sound.wav"
    metadata_path = seg_dir / "metadata.json"

    seg_sound_file = save_wav_file(wav_path, seg_audio_np)
    metadata_path.write_text(json.dumps(dict(speech_seg), indent=2), encoding="utf-8")

    logger.success(f"Segment {seg_number} data saved to:")
    logger.success(seg_sound_file, bright=True)
    logger.success(metadata_path, bright=True)

def _on_transcription_result(
    ja_text: str,
    en_text: str,
    timestamps: list[dict],
    meta: dict,
    overlay: SubtitleOverlay,
    combined_srt_path: Path,
) -> None:
    """Thread-safe callback – called from pipeline worker threads."""
    if not ja_text.strip() or not en_text.strip():
        return

    def add_message_thread() -> None:
        overlay.add_message(
            translated_text=en_text,
            source_text=ja_text,
            start_sec=round(meta["start"], 3),
            end_sec=round(meta["end"], 3),
            duration_sec=round(meta["end"] - meta["start"], 3),
        )
    Thread(target=add_message_thread, daemon=True).start()

    seg_dir = Path(meta["segment_dir"])
    start_sample = meta["start"]
    end_sample = meta["end"]

    write_srt_file(seg_dir / "subtitles.srt", ja_text, en_text, start_sample, end_sample, index=1)
    append_to_combined_srt(
        combined_srt_path,
        ja_text,
        en_text,
        start_sample,
        end_sample,
        index=meta["seg_number"],  # uses the original speech detector idx
    )

def make_result_callback(overlay: SubtitleOverlay, combined_srt_path: Path):
    def callback(
        ja_text: str,
        en_text: str,
        timestamps: list[dict],
        custom_args: dict,
    ) -> None:
        meta = custom_args.get("meta", {})
        if not meta:
            logger.warning("Result callback received without meta – skipping overlay/SRT")
            return

        _on_transcription_result(
            ja_text=ja_text,
            en_text=en_text,
            timestamps=timestamps,
            meta=meta,
            overlay=overlay,
            combined_srt_path=combined_srt_path,
        )
    return callback

if __name__ == "__main__":
    duration_seconds = None
    trim_silent = True
    quit_on_silence = False
    overlap_seconds = 0.5

    app = QApplication([])  # Re-uses existing instance if any
    overlay = SubtitleOverlay.create(app=app, title="Live Japanese Subtitles")
    combined_srt_path = OUTPUT_DIR / "all_subtitles.srt"
    if combined_srt_path.exists():
        combined_srt_path.unlink()
    pipeline = TranscriptionPipeline(max_workers=2, cache_size=500)
    pipeline.on_result = make_result_callback(overlay, combined_srt_path)

    # Run the blocking recording + processing loop in a background thread
    def recording_thread() -> None:
        data_stream = record_from_mic(
            duration_seconds,
            trim_silent=trim_silent,
            quit_on_silence=quit_on_silence,
            overlap_seconds=overlap_seconds,
        )
        segments: list[dict] = []
        for speech_seg, seg_audio_np, full_audio_np in data_stream:
            speech_seg_copy: SpeechSegment = {
                "idx": speech_seg["idx"],
                "start": round(speech_seg["start"] / SAMPLE_RATE, 3),
                "end": round(speech_seg["end"] / SAMPLE_RATE, 3),
                "prob": speech_seg["prob"],
                "duration": speech_seg["duration"],
            }
            seg_number: int = speech_seg["idx"] + 1
            seg_dir = Path(OUTPUT_DIR) / "segments" / f"segment_{seg_number:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            meta: dict = {
                "start": speech_seg_copy["start"],
                "end": speech_seg_copy["end"],
                "segment_dir": str(seg_dir),
                "seg_number": seg_number,
            }
            def pipeline_thread() -> None:
                pipeline.submit_segment(
                    get_wav_bytes(seg_audio_np),
                    meta=meta,
                )
            Thread(target=pipeline_thread, daemon=True).start()
            save_segment_data(speech_seg_copy, seg_audio_np)
            segments.append(dict(speech_seg_copy))
            save_file(segments, OUTPUT_DIR / "all_segments.json", verbose=False)
            output_file = f"{OUTPUT_DIR}/full_recording.wav"
            save_wav_file(output_file, full_audio_np)

    Thread(target=recording_thread, daemon=True).start()

    # Start Qt event loop – this keeps the overlay responsive and visible
    sys.exit(app.exec())