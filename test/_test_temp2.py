# live_subtitles_client.py
"""
Simple example client that:
1. Captures microphone audio (16-bit PCM, 16000 Hz)
2. Sends chunks continuously to the subtitle server
3. Receives and prints live subtitles (Japanese + English translation)

Requirements:
    pip install pyaudio websockets
"""

import asyncio
import base64
import json
import logging
from typing import Optional

import pyaudio
import websockets

# ───────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────

WS_URI = "ws://localhost:8765"          # ← change if server is on different host/port

# Audio settings — must match what server expects
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16                # 16-bit integers
CHUNK_SIZE = 1024                       # samples per chunk (~64 ms @ 16kHz)
BYTES_PER_SAMPLE = 2                    # 16-bit = 2 bytes

# How often to send a chunk (seconds)
SEND_INTERVAL = CHUNK_SIZE / SAMPLE_RATE  # ~0.064 seconds

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("subtitle-client")

# ───────────────────────────────────────────────
# Microphone capture & WebSocket sender
# ───────────────────────────────────────────────

class LiveSubtitleClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.running = False

    async def connect(self):
        logger.info(f"Connecting to {self.uri}...")
        self.websocket = await websockets.connect(self.uri)
        logger.info("Connected!")

    def start_microphone(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        logger.info("Microphone opened")

    def stop_microphone(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        logger.info("Microphone closed")

    async def send_audio_loop(self):
        if not self.stream:
            raise RuntimeError("Microphone not started")

        logger.info("Starting audio streaming... (Ctrl+C to stop)")

        try:
            while self.running:
                try:
                    data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    if len(data) == 0:
                        continue

                    # Encode to base64
                    pcm_b64 = base64.b64encode(data).decode("ascii")

                    payload = {
                        "type": "audio",
                        "pcm": pcm_b64,
                        "sample_rate": SAMPLE_RATE
                    }

                    await self.websocket.send(json.dumps(payload))
                    # logger.debug(f"Sent chunk ({len(data)} bytes)")

                    await asyncio.sleep(SEND_INTERVAL)

                except Exception as e:
                    logger.error(f"Error in send loop: {e}")
                    break

        finally:
            logger.info("Audio send loop stopped")

    async def receive_subtitles_loop(self):
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "subtitle":
                        ja = data.get("transcription_ja", "").strip()
                        en = data.get("translation_en", "").strip()
                        utt_id = data.get("utterance_id")
                        duration = data.get("duration_sec")
                        trigger = data.get("trigger", "unknown")

                        logger.info(
                            f"[#{utt_id}] ({duration:.1f}s) {trigger:<12} | "
                            f"JA: {ja[:80]}{'...' if len(ja) > 80 else ''}"
                        )
                        if en:
                            logger.info(f"          EN: {en}")

                        # Optional: show quality/confidence
                        if "transcription_quality" in data:
                            logger.info(
                                f"          Quality: {data['transcription_quality']} "
                                f"(conf {data.get('transcription_confidence', '?.??'):.2f})"
                            )

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON")
                except Exception as e:
                    logger.error(f"Error processing subtitle: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by server")

    async def run(self):
        self.running = True

        try:
            await self.connect()
            self.start_microphone()

            # Run sender and receiver concurrently
            await asyncio.gather(
                self.send_audio_loop(),
                self.receive_subtitles_loop()
            )

        except KeyboardInterrupt:
            logger.info("Shutting down...")

        except Exception as e:
            logger.exception(f"Client error: {e}")

        finally:
            self.running = False
            self.stop_microphone()
            if self.websocket:
                await self.websocket.close()
            logger.info("Client closed")

# ───────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────

async def main():
    client = LiveSubtitleClient(WS_URI)
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")