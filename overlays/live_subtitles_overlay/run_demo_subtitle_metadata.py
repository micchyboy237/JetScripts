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
    overlay = LiveSubtitlesOverlay.create(title="Live Subtitles – Metadata Demo")

    # Simulated real-world segments – correct keys
    demo_segments = [
        {
            "translated_text": "Hello, how are you today?",
            "source_text": "こんにちは、今日はお元気ですか？",
            "start_sec": 1.250,
            "end_sec": 4.180,
        },
        {
            "translated_text": "I'm doing great, thank you!",
            "source_text": "元気です、ありがとう！",
            "start_sec": 4.500,
            "end_sec": 6.920,
        },
        {
            "translated_text": "That's wonderful to hear.",
            "source_text": "それは素晴らしいですね。",
            "start_sec": 7.100,
            "end_sec": 9.450,
        },
        {
            "translated_text": "What are your plans for the weekend?",
            "source_text": "週末の予定は何ですか？",
            "start_sec": 9.800,
            "end_sec": 12.300,
        },
    ]

    def feed_subtitles() -> None:

        # Start from the first real segment (skip the initial "Ready" message)
        for idx, seg in enumerate(demo_segments, start=2):
            # Natural pacing based on actual segment duration
            time.sleep(seg["end_sec"] - seg["start_sec"] + 0.4)

            overlay.add_message(
                translated_text=seg["translated_text"],
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                duration_sec=round(seg["end_sec"] - seg["start_sec"], 3),
                source_text=seg["source_text"],
            )

            # Print correct metadata table (now shows proper values)
            latest = overlay.message_history[-1]
            from rich.table import Table
            from rich import print as rprint

            table = Table(title=f"Subtitle {idx} metadata")
            table.add_column("Field")
            table.add_column("Value")
            for k, v in latest.items():
                table.add_row(k, str(v))
            rprint(table)

    Thread(target=feed_subtitles, daemon=True).start()

    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_subtitle_metadata()