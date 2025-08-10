import sys

from jet.audio.e2e.send_mic_stream import send_mic_stream


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python send_mic_stream.py <receiver_ip> [port]")
    ip = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    send_mic_stream(ip, port)
