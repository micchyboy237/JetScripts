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
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print(f"✅ Receiving complete. Saved to {OUTPUT_FILE}")
            else:
                print(f"❌ FFmpeg error: {stderr}")
        except KeyboardInterrupt:
            print("🛑 Receiving stopped by user")
            process.terminate()
        except Exception as e:
            print(f"❌ Error during receiving: {str(e)}")
            process.terminate()


if __name__ == "__main__":
    main()
