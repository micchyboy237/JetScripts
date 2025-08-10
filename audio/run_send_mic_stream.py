import argparse
import sys

from jet.audio.e2e.send_mic_stream import send_mic_stream

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send microphone audio via RTP and save segmented WAV files.")
    parser.add_argument("--receiver-ip", type=str, default=None,
                        help="IP address of the RTP receiver (optional)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port number for RTP streaming (optional)")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="Audio sample rate (Hz, default: 44100)")
    parser.add_argument("--channels", type=int, default=2, choices=[
                        1, 2], help="Number of audio channels (1=mono, 2=stereo, default: 2)")
    parser.add_argument("--segment-time", type=int, default=30,
                        help="Duration of each WAV segment in seconds (default: 30)")
    parser.add_argument("--file-prefix", type=str, default="recording",
                        help="Prefix for output WAV files (default: recording)")

    args = parser.parse_args()

    if (args.receiver_ip is None) != (args.port is None):
        parser.error(
            "Both --receiver-ip and --port must be provided together or both omitted")

    send_mic_stream(
        args.receiver_ip,
        args.port,
        args.sample_rate,
        args.channels,
        args.segment_time,
        args.file_prefix
    )
