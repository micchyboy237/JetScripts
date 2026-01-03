import sys
from threading import Thread
import time

from PyQt6.QtWidgets import QApplication
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay


def demo_subtitle_metadata() -> None:
    """
    Demonstrates the new metadata-rich add_message API with realistic live subtitle timing.
    Shows English translation on screen while preserving start/end times, duration, and original Japanese (source) text.
    """
    overlay = LiveSubtitlesOverlay.create(title="Live Subtitles – Full Metadata Demo")

    # Simulated real-world segments – all metadata fields included
    demo_segments = [
        {
            "translated_text": "Hello, how are you today?",
            "source_text": "こんにちは、今日はお元気ですか？",
            "start_sec": 1.25,
            "end_sec": 4.18,
            "duration_sec": 2.93,
            "segment_number": 1,
            "avg_vad_confidence": 0.912,
            "translation_confidence": 0.988,
        },
        {
            "translated_text": "I'm doing great, thank you!",
            # "source_text": "元気です、ありがとう！",
            "start_sec": 4.50,
            "end_sec": 6.92,
            "duration_sec": 2.42,
            "segment_number": 2,
            "avg_vad_confidence": 0.878,
            "translation_confidence": 0.975,
        },
        {
            "translated_text": "That's wonderful to hear.",
            "source_text": "それは素晴らしいですね。",
            "start_sec": 7.10,
            "end_sec": 9.45,
            "duration_sec": 2.35,
            "segment_number": 3,
            "avg_vad_confidence": 0.934,
            "translation_confidence": 0.992,
        },
        {
            "translated_text": "What are your plans for the weekend?",
            # "source_text": "週末の予定は何ですか？",
            "start_sec": 9.80,
            "end_sec": 12.30,
            "duration_sec": 2.50,
            "segment_number": 4,
            "avg_vad_confidence": 0.856,
            "translation_confidence": 0.969,
        },
    ]

    def feed_subtitles() -> None:
        # Start from the first real segment (skip the initial "Ready" message)
        for idx, seg in enumerate(demo_segments, start=1):
            # Faster pacing: fixed short delay, not natural segment duration
            time.sleep(0.25)

            overlay.add_message(
                translated_text=seg["translated_text"],
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                duration_sec=seg["duration_sec"],
                segment_number=seg["segment_number"],
                avg_vad_confidence=seg["avg_vad_confidence"],
                translation_confidence=seg["translation_confidence"],
                # Optional: show original Japanese below translation if present
                source_text=seg.get("source_text"),
            )

            # Print correct metadata table (now shows proper values)
            latest = overlay.message_history[-1]
            from rich.table import Table
            from rich import print as rprint

            table = Table(title=f"Demo Segment {seg['segment_number']} – Expected Metadata")
            table.add_column("Field")
            table.add_column("Value")
            # Show only the fields we care about for clarity
            expected_fields = {
                "translated_text": seg["translated_text"],
                "start_sec": f"{seg['start_sec']:.2f}",
                "end_sec": f"{seg['end_sec']:.2f}",
                "duration_sec": f"{seg['duration_sec']:.2f}",
                "segment_number": str(seg["segment_number"]),
                "avg_vad_confidence": f"{seg['avg_vad_confidence']:.3f}",
                "translation_confidence": f"{seg['translation_confidence']:.3f}",
                "source_text": seg.get("source_text", "(not provided)"),
            }
            for k, v in expected_fields.items():
                table.add_row(k, v)
            rprint(table)

    Thread(target=feed_subtitles, daemon=True).start()

    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_subtitle_metadata()