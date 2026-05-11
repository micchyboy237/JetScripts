import numpy as np
from jet.audio.speech.firered.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from jet.audio.speech.vad_extractors import extract_valley_troughs
from jet.audio.utils.loader import load_audio

if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"

    parser = argparse.ArgumentParser(
        description="Extract valley troughs (strong silence points) from audio or VAD probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options (arguments are now not required to allow default fallback)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--probs",
        "-p",
        type=Path,
        help="Path to .npy file containing VAD probabilities",
    )
    input_group.add_argument(
        "--audio",
        "-a",
        type=Path,
        default=DEFAULT_AUDIO,
        help="Path to audio file (.wav, .mp3, etc.)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=Path(__file__).parent / "generated" / Path(__file__).stem,
        help="Output directory to save JSON results (default: generated/)",
    )

    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # If neither --probs nor --audio is provided, use DEFAULT_AUDIO
        if not getattr(args, "probs", None) and not getattr(args, "audio", None):
            print(
                f"No --audio or --probs provided, using default audio: {DEFAULT_AUDIO}"
            )
            args.audio = Path(DEFAULT_AUDIO)

        if args.probs:
            # Load probabilities from .npy file
            print(f"Loading probabilities from: {args.probs}")
            probs = np.load(args.probs)
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
        else:
            audio = load_audio(args.audio)
            # load_audio can return (audio, sr) tuple
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_np = audio[0]
            else:
                audio_np = audio
            _, probs = extract_speech_timestamps(
                audio=audio_np,
                threshold=0.5,
                min_speech_duration_sec=0.250,
                min_silence_duration_sec=0.250,
                with_scores=True,
            )

        troughs = extract_valley_troughs(
            probs=probs,
            min_valley_duration_s=0.8,
            smoothing_window=20,
            trough_prominence=0.15,
            valley_threshold=None,
            min_trough_offset_s=1.0,
        )

        # Compose output file path
        output_file = args.output_dir / "valley_troughs.json"

        # Output results
        if not troughs:
            print("No valid valley troughs found.")
        else:
            print(f"\nFound {len(troughs)} valley trough(s):\n")

            for i, trough in enumerate(troughs, 1):
                v = trough["valley"]
                print(
                    f"{i:2d}. Time: {trough['time_s']:.3f}s  "
                    f"(Global: {trough.get('global_time_s', trough['time_s']):.3f}s)"
                )
                print(
                    f"    Prob: {trough['prob']:.4f} | "
                    f"Valley Score: {v['valley_score']:.4f} | "
                    f"Trough Score: {v['trough_score']:.4f} | "
                    f"Final Score: {v['final_score']:.4f}"
                )
                print(
                    f"    Duration: {v['duration_s']:.3f}s "
                    f"({v['frame_start']}–{v['frame_end']} frames)\n"
                )

            # Save to JSON if requested
            if args.output_dir:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(troughs, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
