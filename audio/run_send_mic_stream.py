from jet.audio.e2e.send_mic_stream import send_mic_stream
from datetime import datetime
from pathlib import Path

DEFAULT_DEST_IP = "127.0.0.1"
DEFAULT_PORT = "5000"


def main():
    """Main function to demonstrate streaming and saving audio."""
    duration_seconds = 0
    dest_ip = DEFAULT_DEST_IP
    port = DEFAULT_PORT
    output_file = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process = send_mic_stream(
        duration_seconds,
        dest_ip,
        port,
        audio_index="1",
        output_file=str(output_path)
    )
    if process:
        try:
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print(f"✅ Streaming complete, audio saved to {output_file}")
            else:
                print(f"❌ FFmpeg error: {stderr}")
        except KeyboardInterrupt:
            print("🛑 Streaming stopped by user")
            process.terminate()
            if output_path.exists():
                print(f"💾 Audio file saved: {output_file}")
        except Exception as e:
            print(f"❌ Error during streaming: {str(e)}")
            process.terminate()
            if output_path.exists():
                print(f"💾 Audio file saved: {output_file}")


if __name__ == "__main__":
    main()
