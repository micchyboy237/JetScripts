# JetScripts/audio/run_live_subtitles.py (with full overlay message metadata support)

import shutil
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread

from jet.audio.record_mic_speech_detection import record_from_mic
from jet.audio.speech.wav_utils import get_wav_bytes
from jet.audio.transcribers.transcription_pipeline import TranscriptionPipeline
from jet.audio.speech.overlay_utils import write_srt_file, append_to_combined_srt
from jet.overlays.subtitle_overlay import SubtitleOverlay
from jet.file.utils import save_file
from jet.audio.helpers.silence import SAMPLE_RATE

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class RecordingThread(QThread):
    """Run the blocking recording generator in a separate thread to keep Qt responsive."""

    def __init__(self, pipeline: TranscriptionPipeline, combined_srt_path: Path, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.combined_srt_path = combined_srt_path
        self.segments: list[dict] = []
        self.subtitle_index = 1
        # Use sequential index as key to preserve submission order
        self.pending_segments: dict[int, dict] = {}

    def run(self) -> None:
        data_stream = record_from_mic(
            duration=None,  # indefinite – stops on silence or Ctrl+C
            trim_silent=True,
            output_dir=OUTPUT_DIR,
        )

        try:
            for seg_meta, seg_sound_file, seg_audio_np, full_audio_np in data_stream:
                # Convert sample-based times to seconds for JSON metadata
                meta = seg_meta.copy()
                for key in [
                    "original_start_sample",
                    "original_end_sample",
                    "saved_start_sample",
                    "saved_end_sample",
                ]:
                    if key in meta:
                        meta[key] = round(meta[key] / SAMPLE_RATE, 3)

                self.segments.append(meta)
                save_file(self.segments, OUTPUT_DIR / "all_segments.json", verbose=False)

                if seg_audio_np.size > 0:
                    audio_float32 = seg_audio_np.astype(np.float32)
                    # Store meta using the sequential subtitle_index (order of detection)
                    self.pending_segments[self.subtitle_index] = {
                        "meta": meta,
                    }
                    self.subtitle_index += 1
                    self.pipeline.submit_segment(get_wav_bytes(seg_audio_np))

        except Exception as e:
            print(f"\nRecording thread error: {e}")


def _on_transcription_result(
    ja_text: str,
    en_text: str,
    timestamps: list[dict],
    overlay: SubtitleOverlay,
    pending_segments: dict[int, dict],
    combined_srt_path: Path,
):
    """Thread-safe callback – called from pipeline worker threads.

    Now processes completed transcriptions in strict chronological order
    by always consuming the earliest pending index.
    """
    if not ja_text.strip() or not en_text.strip():
        return

    if not pending_segments:
        # Fallback for any stray results (should not happen)
        overlay.add_message(en_text.strip())
        return

    # Always take the lowest (earliest) index still pending
    earliest_index = min(pending_segments.keys())
    info = pending_segments.pop(earliest_index)
    meta = info["meta"]

    start_sec = round(meta["original_start_sample"], 3)
    end_sec = round(meta["original_end_sample"], 3)
    duration_sec = round(end_sec - start_sec, 3)

    overlay.add_message(
        translated_text=en_text,
        source_text=ja_text,
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=duration_sec,
    )

    seg_dir = Path(meta["segment_dir"])
    start_sample = int(meta["original_start_sample"] * SAMPLE_RATE)
    end_sample = int(meta["original_end_sample"] * SAMPLE_RATE)

    write_srt_file(seg_dir / "subtitles.srt", ja_text, en_text, start_sample, end_sample, index=1)
    append_to_combined_srt(
        combined_srt_path,
        ja_text,
        en_text,
        start_sample,
        end_sample,
        index=earliest_index,  # guaranteed sequential
    )


if __name__ == "__main__":
    # Qt app and overlay (must be created on main thread)
    app = QApplication([])  # Re-uses existing instance if any
    overlay = SubtitleOverlay.create(app=app, title="Live Japanese Subtitles")

    # Combined SRT file
    combined_srt_path = OUTPUT_DIR / "all_subtitles.srt"
    if combined_srt_path.exists():
        combined_srt_path.unlink()

    # Transcription pipeline
    pipeline = TranscriptionPipeline(max_workers=2, cache_size=500)

    # Recording thread
    recording_thread = RecordingThread(pipeline=pipeline, combined_srt_path=combined_srt_path)
    # Bind callback with required shared objects
    pipeline.on_result = lambda ja, en, ts: _on_transcription_result(
        ja, en, ts, overlay, recording_thread.pending_segments, combined_srt_path
    )

    # Start recording in background
    recording_thread.start()

    # Run Qt event loop – overlay stays responsive, subtitles appear live
    try:
        app.exec()
    finally:
        pipeline.shutdown(wait=True)
        recording_thread.wait(timeout=5)  # Graceful cleanup