# run_live_subtitles.py
import json
from pathlib import Path
import shutil
import sys
from threading import Thread
from typing import TypedDict
from PyQt6.QtWidgets import QApplication
import numpy as np

from jet.audio.helpers.silence import SAMPLE_RATE
from jet.audio.record_mic_speech_detection import record_from_mic
from jet.audio.speech.output_utils import append_to_combined_srt, write_srt_file
from jet.audio.speech.silero.speech_types import SpeechSegment
from jet.audio.speech.wav_utils import get_wav_bytes, save_wav_file
from jet.audio.transcribers.base import AudioInput
from jet.audio.transcribers.transcription_pipeline import transcribe_ja_chunk
from jet.audio.audio_search import AudioSegmentDatabase
from jet.file.utils import save_file
from jet.logger import logger
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay, SubtitleMessage

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
# Persistent audio search database for caching embeddings + subtitles
DB_DIR = str(OUTPUT_DIR / "audio_vector_db")
RESULTS_DIR = OUTPUT_DIR / "results"
shutil.rmtree(RESULTS_DIR, ignore_errors=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

audio_db = AudioSegmentDatabase(persist_dir=DB_DIR)
SIMILARITY_THRESHOLD = 0.95  # Reuse subtitles if audio similarity >= this

class SubtitlesMeta(TypedDict):
    start: float
    end: float
    segment_dir: str
    seg_number: int


def save_segment_data(speech_seg: SpeechSegment, seg_audio_np: np.ndarray):
    segment_root = Path(RESULTS_DIR) / "segments"
    segment_root.mkdir(parents=True, exist_ok=True)

    # Use the canonical num from the speech detector as segment number
    seg_number: int = speech_seg["num"]
    seg_dir = segment_root / f"segment_{seg_number:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    wav_path = seg_dir / "sound.wav"
    metadata_path = seg_dir / "metadata.json"

    seg_sound_file = save_wav_file(wav_path, seg_audio_np)
    metadata_path.write_text(json.dumps(dict(speech_seg), indent=2), encoding="utf-8")

    logger.success(f"Segment {seg_number} data saved to:")
    logger.success(seg_sound_file, bright=True)
    logger.success(metadata_path, bright=True)

def handle_transcription_result(
    ja_text: str,
    en_text: str,
    timestamps: list[dict],
    meta: SubtitlesMeta,
    combined_srt_path: Path,
) -> None:
    """Write subtitles .srt files"""

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
        index=meta["seg_number"],
    )

async def translation_process(audio: AudioInput, meta: SubtitlesMeta) -> SubtitleMessage:
    result = await transcribe_ja_chunk(audio)
    logger.debug("transcribe_audio returned: %r", result)

    ja_text = result["transcription"].strip()
    en_text = result.get("translation", "").strip()
    timestamps = result.get("words", result.get("segments", []))

    return {
        "source_text": ja_text,
        "translated_text": en_text,
        "start_sec": round(meta["start"], 3),
        "end_sec": round(meta["end"], 3),
        "duration_sec": round(meta["end"] - meta["start"], 3),
    }

if __name__ == "__main__":
    duration_seconds = None
    trim_silent = False
    quit_on_silence = False
    overlap_seconds = 0.5

    app = QApplication([])  # Re-uses existing instance if any
    overlay = LiveSubtitlesOverlay.create(app=app, title="Live Japanese Subtitles")
    combined_srt_path = RESULTS_DIR / "all_subtitles.srt"
    if combined_srt_path.exists():
        combined_srt_path.unlink()
    # pipeline = TranscriptionPipeline(max_workers=2, cache_size=500)
    # pipeline.on_result = lambda ja, en, ts, args: handle_transcription_result(
    #     ja, en, ts, args, overlay, combined_srt_path
    # )

    # Run the blocking recording + processing loop in a background thread
    def recording_thread() -> None:
        data_stream = record_from_mic(
            duration_seconds,
            trim_silent=trim_silent,
            quit_on_silence=quit_on_silence,
            overlap_seconds=overlap_seconds,
        )
        segments: list[SpeechSegment] = []
        for speech_seg, seg_audio_np, full_audio_np in data_stream:
            speech_seg_meta: SpeechSegment = speech_seg.copy()
            speech_seg_meta["start"] = round(speech_seg["start"] / SAMPLE_RATE, 3)
            speech_seg_meta["end"] = round(speech_seg["end"] / SAMPLE_RATE, 3)
            seg_number: int = speech_seg_meta["num"]
            seg_dir = Path(RESULTS_DIR) / "segments" / f"segment_{seg_number:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            meta: SubtitlesMeta = {
                "start": speech_seg_meta["start"],
                "end": speech_seg_meta["end"],
                "segment_dir": str(seg_dir),
                "seg_number": seg_number,
            }
            # Save segment first (needed for potential search and indexing)
            save_segment_data(speech_seg_meta, seg_audio_np)


            def pipeline_thread() -> None:
                wav_bytes = get_wav_bytes(seg_audio_np)

                # Search for similar existing segment before transcribing
                similar = audio_db.search_similar(wav_bytes, top_k=1)

                cache_hit = False
                ja_text = ""
                en_text = ""
                timestamps: list[dict] = []

                if similar and similar[0]["score"] >= SIMILARITY_THRESHOLD:
                    match = similar[0]
                    ja_text = match.get("ja_text", "").strip()
                    en_text = match.get("en_text", "").strip()
                    logger.success(
                        f"Audio cache hit! Reusing subtitles (score={match['score']:.4f}) "
                        f"from segment {match.get('seg_number', 'unknown')}"
                    )
                    cache_hit = True

                if cache_hit:
                    # Cache hit → immediately process result (synchronous)
                    handle_transcription_result(
                        ja_text, en_text, timestamps, meta,
                        combined_srt_path=combined_srt_path
                    )
                else:
                    # Cache miss → launch async transcription + handler
                    async def process_and_handle() -> SubtitleMessage:
                        result_msg = await translation_process(
                            audio=wav_bytes, meta=meta
                        )
                        handle_transcription_result(
                            ja_text=result_msg["source_text"],
                            en_text=result_msg["translated_text"],
                            timestamps=timestamps,  # still empty for now
                            meta=meta,
                            combined_srt_path=combined_srt_path,
                        )
                        return result_msg

                    overlay.add_task(process_and_handle)

            Thread(target=pipeline_thread, daemon=True).start()

            segments.append(speech_seg_meta)
            save_file(segments, RESULTS_DIR / "all_segments.json", verbose=False)
            output_file = f"{RESULTS_DIR}/full_recording.wav"
            save_wav_file(output_file, full_audio_np)

    Thread(target=recording_thread, daemon=True).start()

    # Start Qt event loop – this keeps the overlay responsive and visible
    sys.exit(app.exec())