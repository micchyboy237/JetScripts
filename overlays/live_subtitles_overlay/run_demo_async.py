import asyncio
import sys

from PyQt6.QtWidgets import QApplication
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay


# Async tasks demo
def demo_async():
    async def long_running_translation(text: str) -> str:
        await asyncio.sleep(1.0)
        return f"Translated: {text.upper()}"

    overlay = LiveSubtitlesOverlay.create(title="Async Demo")
    overlay.add_task(long_running_translation, "hello world")
    overlay.add_task(long_running_translation, "first")
    overlay.add_task(long_running_translation, "second")
    overlay.add_task(long_running_translation, "third")
    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_async()