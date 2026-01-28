import sys
from threading import Thread

from PyQt6.QtWidgets import QApplication
from jet.logger.timer import sleep_countdown
from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay

def demo_threading():
    overlay = LiveSubtitlesOverlay.create(title="Threading Demo")

    def demo():
        msg_id = overlay.add_message(
            translated_text="Hello—",
            segment_number=42,
        )
        sleep_countdown(3)
        overlay.update_message(
            msg_id,
            translated_text="Hello, how are you today?",
            transcription_confidence=0.98,
            transcription_quality="High",
        )

    Thread(target=demo, daemon=True).start()
    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_threading()