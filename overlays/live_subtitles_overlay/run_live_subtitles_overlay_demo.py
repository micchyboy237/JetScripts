# run_demo_subtitle_metadata.py
import sys
import time
from threading import Thread

from jet.overlays.live_subtitles_overlay import LiveSubtitlesOverlay
from PyQt6.QtWidgets import QApplication


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
        title="Live Subtitles – Full Metadata Demo (2026)",
        on_clear=lambda: print("[DEMO] Clear operation completed successfully ✓"),
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
            "translation_confidence": 0.947,  # normalized 0.0–1.0
            "translation_quality": "Very High",
        },
        {
            "translated_text": (
                "The quick brown fox jumps over the lazy dog multiple times while exploring the vast forest "
                "near the quiet mountain village where ancient trees whisper secrets to anyone willing to "
                "listen carefully on a peaceful autumn afternoon filled with golden sunlight filtering through "
                "colorful leaves slowly drifting to the soft moss-covered ground creating a magical carpet that "
                "invites quiet reflection and deep thoughts about life nature and the passing of seasons in "
                "harmony with the gentle wind that carries distant bird songs across the valley below where "
                "a small clear river flows steadily past old stone bridges and wildflower meadows buzzing with "
                "bees and butterflies dancing in the warm breeze under an endless blue sky dotted with fluffy "
                "white clouds that slowly change shape as the day progresses toward a beautiful sunset painting "
                "the horizon in shades of orange pink and purple promising another peaceful night under the stars."
            ),
            "source_text": (
                "素早い茶色のキツネが怠け者の犬を何度も飛び越えながら、静かな山間の村の近くにある広大な森を探検しています。"
                "そこで古い木々が、耳を傾ける気のある人々に秘密を囁き、穏やかな秋の午後に金色の陽光が色とりどりの葉を通して"
                "ゆっくりと降り注ぎ、柔らかい苔に覆われた地面にゆっくりと落ちて魔法のような絨毯を作り出し、人生や自然、"
                "そして季節の移り変わりについて静かに考えることを誘います。優しい風が谷の下を流れる小さな澄んだ川のそばを"
                "通り過ぎ、古い石橋や野花の咲く草原を横切り、ミツバチや蝶が暖かいそよ風の中で踊る様子を運び、無限に広がる"
                "青い空に浮かぶふわふわした白い雲がゆっくりと形を変えながら、美しい夕焼けへと一日が進み、オレンジ、ピンク、"
                "紫の色で地平線を染め、星空の下でまた穏やかな夜を約束しています。"
            ),
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
        {
            "translated_text": "I plan to visit my family in the countryside.",
            "source_text": "田舎の家族を訪ねるつもりです。",
            "start_sec": 12.80,
            "end_sec": 15.60,
            "duration_sec": 2.80,
            "segment_number": 5,
            "avg_vad_confidence": 0.845,
            "transcription_confidence": 0.981,
            "transcription_quality": "High",
            "translation_confidence": 0.923,
            "translation_quality": "High",
        },
        {
            "translated_text": "That sounds relaxing! Will you go hiking too?",
            "source_text": "リラックスできそう！ハイキングもするの？",
            "start_sec": 15.90,
            "end_sec": 18.70,
            "duration_sec": 2.80,
            "segment_number": 6,
            "avg_vad_confidence": 0.678,
            "transcription_confidence": 0.962,
            "transcription_quality": "Good",
            "translation_confidence": 0.875,
            "translation_quality": "Good",
        },
        {
            "translated_text": "Maybe. I also want to read some books and enjoy the quiet.",
            # source_text omitted to demonstrate fallback
            "start_sec": 19.00,
            "end_sec": 22.50,
            "duration_sec": 3.50,
            "segment_number": 7,
            "avg_vad_confidence": 0.892,
            "transcription_confidence": 0.990,
            "transcription_quality": "Very High",
            "translation_confidence": 0.958,
            "translation_quality": "Very High",
        },
        {
            "translated_text": "Perfect way to recharge. I hope you have a wonderful time!",
            "source_text": "最高のリフレッシュ方法だね。素敵な時間を過ごしてね！",
            "start_sec": 23.10,
            "end_sec": 26.80,
            "duration_sec": 3.70,
            "segment_number": 8,
            "avg_vad_confidence": 0.733,
            "transcription_confidence": 0.977,
            "transcription_quality": "High",
            "translation_confidence": 0.901,
            "translation_quality": "High",
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
            from rich import print as rprint
            from rich.table import Table

            table = Table(title=f"Demo Segment {seg['segment_number']} – Sent Metadata")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("translated_text", seg["translated_text"])
            table.add_row("source_text", seg.get("source_text", "[none]"))
            table.add_row("start_sec", f"{seg['start_sec']:.2f}")
            table.add_row("end_sec", f"{seg['end_sec']:.2f}")
            table.add_row("duration_sec", f"{seg['duration_sec']:.2f}")
            table.add_row("segment_number", str(seg["segment_number"]))
            table.add_row(
                "avg_vad_confidence", f"{seg.get('avg_vad_confidence', '—'):.3f}"
            )
            table.add_row(
                "transcription_confidence",
                f"{seg.get('transcription_confidence', '—'):.3f}",
            )
            table.add_row(
                "transcription_quality", seg.get("transcription_quality", "—")
            )
            table.add_row(
                "translation_confidence",
                f"{seg.get('translation_confidence', '—'):.3f}",
            )
            table.add_row("translation_quality", seg.get("translation_quality", "—"))

            rprint(table)
            print()  # small visual separation

    # Launch the feeder in background thread
    Thread(target=feed_subtitles, daemon=True).start()

    # Start Qt event loop
    sys.exit(QApplication.exec())


if __name__ == "__main__":
    demo_subtitle_metadata()
