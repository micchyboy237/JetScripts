import subprocess

from datetime import datetime
from pathlib import Path

from jet.audio.record_mic_stream import record_mic_stream

import sys
import threading
import time

OUTPUT_DIR = Path(__file__).parent / "generated" / "run_record_mic_stream"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"


def run_record_mic_stream(duration_seconds: int = 60):
    """Main function to demonstrate recording."""

    def print_countdown(total: int):
        """Print live countdown timer in place."""
        for remaining in range(total, -1, -1):
            mins, secs = divmod(remaining, 60)
            # Show "Recording... 0:34 remaining" format
            sys.stdout.write(f"\r🎙️  Recording... {mins}:{secs:02d} remaining  ")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r🎙️  Recording... complete!       \n")
        sys.stdout.flush()

    # Start countdown in background thread
    timer_thread = threading.Thread(target=print_countdown, args=(duration_seconds,), daemon=True)
    timer_thread.start()

    process = record_mic_stream(duration_seconds, OUTPUT_FILE, audio_index="1")

    if process:
        try:
            # Wait for the recording to complete
            stdout, stderr = process.communicate(timeout=duration_seconds + 2)
            if process.returncode == 0:
                # Final success message (countdown already says "complete!")
                print(f"✅ Recording complete. Saved to {OUTPUT_FILE}")
            else:
                print(f"❌ FFmpeg error: {stderr}")
        except subprocess.TimeoutExpired:
            process.terminate()
            sys.stdout.write("\r❌ Recording timed out! Force stopping...       \n")
        except Exception as e:
            print(f"❌ Error during recording: {str(e)}")
            process.terminate()
    else:
        # If recording failed to start, stop countdown visual
        sys.stdout.write("\r❌ Failed to start recording.                 \n")


if __name__ == "__main__":
    duration_seconds = 70
    run_record_mic_stream(duration_seconds)
