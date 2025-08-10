import time
import subprocess

from datetime import datetime
from pathlib import Path

from jet.audio.e2e.receive_mic_stream import receive_mic_stream

OUTPUT_DIR = Path(__file__).parent / "generated" / "run_receive_mic_stream"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / \
    f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
DEFAULT_LISTEN_IP = "0.0.0.0"  # Listen on all interfaces
DEFAULT_PORT = "5000"


def main():
    """Main function to demonstrate receiving."""
    listen_ip = DEFAULT_LISTEN_IP
    port = DEFAULT_PORT
    process = receive_mic_stream(OUTPUT_FILE, listen_ip, port)
    if process:
        try:
            while process.poll() is None:
                time.sleep(1)  # Keep the process running until interrupted
        except KeyboardInterrupt:
            print("🛑 Receiving stopped by user")
            process.terminate()
            try:
                process.wait(timeout=5)  # Wait for FFmpeg to cleanly exit
            except subprocess.TimeoutExpired:
                print("⚠️ FFmpeg did not terminate cleanly, forcing kill")
                process.kill()
            if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
                print(
                    f"✅ Receiving complete. Saved to {OUTPUT_FILE}, size: {OUTPUT_FILE.stat().st_size} bytes")
            else:
                print(f"❌ Output file not created or empty: {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Error during receiving: {str(e)}")
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
