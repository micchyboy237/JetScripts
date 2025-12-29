import asyncio
import sys

from PyQt6.QtWidgets import QApplication
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay
from jet.overlays.live_subtitles_overlay import SubtitleMessage

# Async tasks demo
def demo_async():
    async def long_running_translation(text: str) -> SubtitleMessage:
        await asyncio.sleep(1.0)
        return {
            "translated_text": f"Translated: {text.upper()}",
            "source_text": text,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "duration_sec": 1.0,
        }

    overlay = LiveSubtitlesOverlay.create(title="Async Demo")
    overlay.add_task(long_running_translation, "hello world")
    overlay.add_task(long_running_translation, "good morning")
    overlay.add_task(long_running_translation, "how are you?")
    overlay.add_task(long_running_translation, "thank you very much")
    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_async()