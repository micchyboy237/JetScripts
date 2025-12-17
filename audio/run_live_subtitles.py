# JetScripts/audio/run_live_subtitles.py (full updated file – fixed for parallel overlay + SRT saving)

import shutil
from pathlib import Path
from typing import Dict

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread

from jet.audio.record_mic_speech_detection import record_from_mic
from jet.audio.transcribers.transcription_pipeline import TranscriptionPipeline, AudioKey
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
        self.pending_segments: Dict[AudioKey, dict] = {}

    def run(self) -> None:
        data_stream = record_from_mic(
            duration=None,  # indefinite – stops on silence or Ctrl+C
            trim_silent=True,
            output_dir=OUTPUT_DIR,
        )

        try:
            for seg_meta, seg_audio_np, full_audio_np in data_stream:
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
                    audio_float32 = seg_audio_np.astype("float32")
                    key = self.pipeline._make_key(audio_float32)
                    self.pending_segments[key] = {
                        "meta": meta,
                        "index": self.subtitle_index,
                    }
                    self.subtitle_index += 1
                    self.pipeline.submit_segment(audio_float32)

        except Exception as e:
            print(f"\nRecording thread error: {e}")


def _on_transcription_result(
    ja_text: str,
    en_text: str,
    timestamps: list[dict],
    overlay: SubtitleOverlay,
    pending_segments: Dict[AudioKey, dict],
    combined_srt_path: Path,
):
    """Thread-safe callback – called from pipeline worker threads."""
    if not ja_text.strip():
        return

    overlay.add_message(ja_text.strip())

    # Match result to the most recent pending segment (sequential submission → reliable)
    if pending_segments:
        # Largest duration ≈ latest segment
        key = max(pending_segments.keys(), key=lambda k: k.duration_sec)
        info = pending_segments.pop(key)
        meta = info["meta"]
        idx = info["index"]

        seg_dir = Path(meta["segment_dir"])
        start_sample = int(meta["original_start_sample"] * SAMPLE_RATE)
        end_sample = int(meta["original_end_sample"] * SAMPLE_RATE)

        # Per-segment SRT (single entry)
        write_srt_file(seg_dir / "subtitles.srt", ja_text, start_sample, end_sample, index=1)

        # Append to combined SRT
        append_to_combined_srt(combined_srt_path, ja_text, start_sample, end_sample, index=idx)


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