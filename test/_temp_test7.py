from pathlib import Path
from typing import Dict, Optional

import requests

# ==================== CONFIG ====================
SERVER_URL = "http://192.168.68.150:8000"


def read_audio_bytes(file_path: str) -> bytes:
    """Read audio file as raw bytes"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    audio_bytes = path.read_bytes()
    print(f"Loaded audio file: {len(audio_bytes):,} bytes")
    return audio_bytes


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[Dict]:
    """Call /transcribe endpoint"""
    print("\n🎤 Sending audio to /transcribe ...")

    files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"sample_rate": str(sample_rate)}

    response = requests.post(
        f"{SERVER_URL}/transcribe", files=files, data=data, timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        ja_text = result.get("transcription_ja", "").strip()

        print("✅ Transcription successful!")
        print(f"Japanese: {ja_text}")

        if not ja_text:
            print("⚠️  No transcription text returned.")

        return result
    else:
        print(f"❌ Transcription failed: {response.status_code}")
        print(response.text)
        return None


def translate_text(japanese_text: str, temperature: float = 0.35) -> Optional[Dict]:
    """Call /translate endpoint"""
    if not japanese_text:
        print("⚠️  Empty Japanese text, skipping translation.")
        return None

    print(
        f"\n🌐 Sending to /translate: {japanese_text[:80]}{'...' if len(japanese_text) > 80 else ''}"
    )

    payload = {"japanese_text": japanese_text, "temperature": temperature}

    response = requests.post(f"{SERVER_URL}/translate", json=payload, timeout=15)

    if response.status_code == 200:
        result = response.json()
        print("✅ Translation successful!")
        print(f"English : {result.get('translation_en')}")
        print(f"Quality : {result.get('quality')}")
        return result
    else:
        print(f"❌ Translation failed: {response.status_code}")
        print(response.text)
        return None


def transcribe_and_translate(
    audio_path: str, sample_rate: int = 16000, temperature: float = 0.35
):
    """Complete pipeline: Transcribe → Translate"""
    try:
        # Step 1: Read audio bytes
        audio_bytes = read_audio_bytes(audio_path)

        # Step 2: Transcribe
        trans_result = transcribe_audio(audio_bytes, sample_rate)

        if not trans_result:
            return None

        ja_text = trans_result.get("transcription_ja", "").strip()

        # Step 3: Translate the transcription result
        if ja_text:
            trans_result["translation"] = translate_text(ja_text, temperature)
        else:
            print("⚠️  Skipping translation because transcription returned empty text.")

        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE")
        print("=" * 60)

        return trans_result

    except Exception as e:
        print(f"❌ Client error: {e}")
        return None


# ==================== RUN ====================
if __name__ == "__main__":
    import argparse

    DEFAULT_AUDIO_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/audio_waveform/generated/speech_tracking/saved_speech_segments/segment_20260502_050846_463/sound.wav"

    parser = argparse.ArgumentParser(
        description="Transcribe a Japanese audio segment and translate it to English."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO_PATH,
        type=str,
        help=f"Path to the audio file (default: {DEFAULT_AUDIO_PATH})",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.35,
        help="Temperature for translation model (default: 0.35)",
    )
    args = parser.parse_args()

    result = transcribe_and_translate(
        audio_path=args.audio_path,
        sample_rate=args.sample_rate,
        temperature=args.temperature,
    )

    # Optional: Pretty print final result
    if result:
        import json

        print("\nFinal combined result:")
        print(
            json.dumps(
                {
                    "transcription_ja": result.get("transcription_ja"),
                    "translation_en": result.get("translation", {}).get(
                        "translation_en"
                    )
                    if result.get("translation")
                    else None,
                    "metadata": result.get("metadata"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
