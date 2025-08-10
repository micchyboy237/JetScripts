import sys

from jet.audio.e2e.receive_mic_stream import receive_stream


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    receive_stream(port)
