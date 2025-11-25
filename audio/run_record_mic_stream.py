import subprocess
import shutil

from datetime import datetime
from pathlib import Path

from jet.audio.record_mic_stream import record_mic_stream

OUTPUT_DIR = Path(__file__).parent / "generated" / "run_record_mic_stream"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"


def main():
    """Main function to demonstrate recording."""
    duration_seconds = 5
    process = record_mic_stream(duration_seconds, OUTPUT_FILE, audio_index="1")

    if process:
        try:
            # Wait for the recording to complete
            stdout, stderr = process.communicate(timeout=duration_seconds + 2)
            if process.returncode == 0:
                print(f"✅ Recording complete. Saved to {OUTPUT_FILE}")
            else:
                print(f"❌ FFmpeg error: {stderr}")
        except subprocess.TimeoutExpired:
            process.terminate()
            print("❌ Recording timed out")
        except Exception as e:
            print(f"❌ Error during recording: {str(e)}")
            process.terminate()


if __name__ == "__main__":
    main()
