import asyncio
import base64
import json

import sounddevice as sd
import websockets
from pysilero_vad import SileroVoiceActivityDetector

# =============================
# Configuration
# =============================

WS_URL = "ws://192.168.68.150:8765"

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

VAD_THRESHOLD = 0.5  # speech probability threshold

# =============================
# Initialize Silero VAD
# =============================

vad = SileroVoiceActivityDetector()  # model_path uses default

VAD_CHUNK_SAMPLES = vad.chunk_samples()
VAD_CHUNK_BYTES = vad.chunk_bytes()

print(f"[VAD] chunk_samples={VAD_CHUNK_SAMPLES}, chunk_bytes={VAD_CHUNK_BYTES}")

# =============================
# Audio capture + streaming
# =============================

async def stream_microphone(ws: websockets.WebSocketClientProtocol) -> None:
    """
    Capture mic audio, apply VAD, send speech-only PCM frames.
    """
    pcm_buffer = bytearray()

    def audio_callback(indata, frames: int, time, status) -> None:
        nonlocal pcm_buffer
        if status:
            print(status)
        # indata is a buffer object -> copy to owned bytes
        pcm_buffer.extend(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=VAD_CHUNK_SAMPLES,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
    ):
        print("🎙️  Microphone streaming started")

        try:
            while True:
                await asyncio.sleep(0.01)

                while len(pcm_buffer) >= VAD_CHUNK_BYTES:
                    chunk = bytes(pcm_buffer[:VAD_CHUNK_BYTES])
                    del pcm_buffer[:VAD_CHUNK_BYTES]

                    speech_prob: float = vad(chunk)

                    if speech_prob < VAD_THRESHOLD:
                        continue  # skip silence

                    payload = {
                        "type": "audio",
                        "sample_rate": SAMPLE_RATE,
                        "pcm": base64.b64encode(chunk).decode("ascii"),
                    }

                    asyncio.create_task(ws.send(json.dumps(payload)))
        except asyncio.CancelledError:
            print("\n🛑 Streaming task cancelled")
            raise
        finally:
            print("🎙️  Microphone streaming stopped")


async def receive_subtitles(ws) -> None:
    """
    Receive partial/final subtitles from server.
    """
    async for msg in ws:
        data = json.loads(msg)
        if data.get("type") == "subtitle":
            ja = data.get("transcription", "")
            en = data.get("translation", "")
            print(f"JA: {ja}\nEN: {en}\n")


# =============================
# Main entrypoint
# =============================

async def main() -> None:
    try:
        async with websockets.connect(
            WS_URL,
            max_size=None,
            compression=None,
        ) as ws:
            print(f"Connected to {WS_URL}")
            await asyncio.gather(
                stream_microphone(ws),
                receive_subtitles(ws),
            )
    except websockets.ConnectionClosedOK:
        print("\nConnection closed normally by server")
    except websockets.ConnectionClosedError:
        print("\nConnection closed abnormally")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("Client shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
