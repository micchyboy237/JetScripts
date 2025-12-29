import asyncio
import sys

from PyQt6.QtWidgets import QApplication
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay
from jet.overlays.live_subtitles_overlay import SubtitleMessage

def demo_async():
    async def long_running_translation(
        translated_text: str,
        source_text: str,
        start_sec: float,
        end_sec: float,
        duration_sec: float,
    ) -> SubtitleMessage:
        await asyncio.sleep(1.0)
        # In a real app, this would call a translator API and compute timing
        # Here we just echo back the provided metadata with minor formatting
        return {
            "translated_text": translated_text,
            "source_text": source_text,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
        }

    overlay = LiveSubtitlesOverlay.create(title="Async Demo – Keyword Args")

    # Each add_task passes full subtitle metadata as individual keyword arguments
    overlay.add_task(
        long_running_translation,
        translated_text="Hello, how are you today?",
        source_text="こんにちは、今日はお元気ですか？",
        start_sec=1.0,
        end_sec=4.2,
        duration_sec=3.2,
    )
    overlay.add_task(
        long_running_translation,
        translated_text="Thank you very much for your help.",
        source_text="Merci beaucoup pour votre aide.",
        start_sec=5.0,
        end_sec=8.5,
        duration_sec=3.5,
    )
    overlay.add_task(
        long_running_translation,
        translated_text="I hope you're doing well today.",
        source_text="¿Cómo estás hoy? Espero que bien.",
        start_sec=9.0,
        end_sec=13.8,
        duration_sec=4.8,
    )
    overlay.add_task(
        long_running_translation,
        translated_text="Thanks for the quick response!",
        source_text="Vielen Dank für die schnelle Antwort!",
        start_sec=14.5,
        end_sec=17.0,
        duration_sec=2.5,
    )

    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_async()