from jet.audio.e2e.send_mic_stream import send_mic_stream


DEFAULT_DEST_IP = "127.0.0.1"  # Change to receiver's LAN IP for different devices
DEFAULT_PORT = "5000"


def main():
    """Main function to demonstrate streaming."""
    duration_seconds = 0  # Stream indefinitely until stopped
    dest_ip = DEFAULT_DEST_IP
    port = DEFAULT_PORT
    process = send_mic_stream(duration_seconds, dest_ip, port, audio_index="1")

    if process:
        try:
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print("✅ Streaming complete")
            else:
                print(f"❌ FFmpeg error: {stderr}")
        except KeyboardInterrupt:
            print("🛑 Streaming stopped by user")
            process.terminate()
        except Exception as e:
            print(f"❌ Error during streaming: {str(e)}")
            process.terminate()


if __name__ == "__main__":
    main()
