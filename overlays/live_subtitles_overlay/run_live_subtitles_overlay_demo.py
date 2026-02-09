# run_demo_subtitle_metadata.py
import sys
from threading import Thread
import time
from PyQt6.QtWidgets import QApplication
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay


def demo_subtitle_metadata() -> None:
    """
    Demonstrates the full metadata-rich add_message API with realistic values.
    Shows:
    - English translation (main text)
    - Japanese source text (when available)
    - Segment number
    - Timing (start/end/duration)
    - VAD confidence
    - Transcription confidence + quality label
    - Translation confidence (0.0–1.0 normalized) + quality label
    """
    overlay = LiveSubtitlesOverlay.create(
        title="Live Subtitles – Full Metadata Demo (2026)"
    )

    # Simulated real-world subtitle segments with all supported fields
    demo_segments = [
        {
            "translated_text": "Hello, how are you today?",
            "source_text": "こんにちは、今日はお元気ですか？",
            "start_sec": 1.25,
            "end_sec": 4.18,
            "duration_sec": 2.93,
            "segment_number": 1,
            "avg_vad_confidence": 0.912,
            "transcription_confidence": 0.988,
            "transcription_quality": "High",
            "translation_confidence": 0.947,          # normalized 0.0–1.0
            "translation_quality": "Very High",
        },
        {
            "translated_text": "I'm doing great, thank you!",
            # source_text omitted on purpose (to show graceful fallback)
            "start_sec": 4.50,
            "end_sec": 6.92,
            "duration_sec": 2.42,
            "segment_number": 2,
            "avg_vad_confidence": 0.500,
            "transcription_confidence": 0.975,
            "transcription_quality": "Good",
            "translation_confidence": 0.892,
            "translation_quality": "Good",
        },
        {
            "translated_text": "That's wonderful to hear.",
            "source_text": "それは素晴らしいですね。",
            "start_sec": 7.10,
            "end_sec": 9.45,
            "duration_sec": 2.35,
            "segment_number": 3,
            "avg_vad_confidence": 0.325,
            "transcription_confidence": 0.992,
            "transcription_quality": "Very High",
            "translation_confidence": 0.971,
            "translation_quality": "Very High",
        },
        {
            "translated_text": "What are your plans for the weekend?",
            "source_text": "週末の予定は何ですか？",
            "start_sec": 9.80,
            "end_sec": 12.30,
            "duration_sec": 2.50,
            "segment_number": 4,
            "avg_vad_confidence": 0.756,
            "transcription_confidence": 0.969,
            "transcription_quality": "Medium",
            "translation_confidence": 0.814,
            "translation_quality": "Medium",
        },
    ]

    def feed_subtitles() -> None:
        # Give overlay a moment to appear
        time.sleep(0.8)

        for seg in demo_segments:
            time.sleep(1.2)  # realistic-ish pacing for demo visibility

            overlay.add_message(
                translated_text=seg["translated_text"],
                source_text=seg.get("source_text"),
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                duration_sec=seg["duration_sec"],
                segment_number=seg["segment_number"],
                avg_vad_confidence=seg.get("avg_vad_confidence"),
                transcription_confidence=seg.get("transcription_confidence"),
                transcription_quality=seg.get("transcription_quality"),
                translation_confidence=seg.get("translation_confidence"),
                translation_quality=seg.get("translation_quality"),
            )

            # Print verification table with rich
            from rich.table import Table
            from rich import print as rprint

            table = Table(title=f"Demo Segment {seg['segment_number']} – Sent Metadata")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("translated_text", seg["translated_text"])
            table.add_row("source_text", seg.get("source_text", "[none]"))
            table.add_row("start_sec", f"{seg['start_sec']:.2f}")
            table.add_row("end_sec", f"{seg['end_sec']:.2f}")
            table.add_row("duration_sec", f"{seg['duration_sec']:.2f}")
            table.add_row("segment_number", str(seg["segment_number"]))
            table.add_row("avg_vad_confidence", f"{seg.get('avg_vad_confidence', '—'):.3f}")
            table.add_row("transcription_confidence", f"{seg.get('transcription_confidence', '—'):.3f}")
            table.add_row("transcription_quality", seg.get("transcription_quality", "—"))
            table.add_row("translation_confidence", f"{seg.get('translation_confidence', '—'):.3f}")
            table.add_row("translation_quality", seg.get("translation_quality", "—"))

            rprint(table)
            print()  # small visual separation

    # Launch the feeder in background thread
    Thread(target=feed_subtitles, daemon=True).start()

    # Start Qt event loop
    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_subtitle_metadata()