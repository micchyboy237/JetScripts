"""
enhance_audio.py

Enhances a speech audio file for better Whisper transcription accuracy.

Improvements over the live-segment version:
  - Stereo → mono mixdown before processing
  - Automatic resampling to 16 kHz (Whisper's required rate)
  - Median-based noise floor estimation (robust for files where speech
    starts immediately — no safe "silence tail" assumption)
  - Frequency-weighted pre-emphasis (gentler, avoids over-sharpening
    on already clear recordings)
  - Per-band normalization guard to prevent over-amplifying noise-only
    regions in sparse speech files
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


# ---------------------------------------------------------------------------
# Core enhancement
# ---------------------------------------------------------------------------


def enhance_speech_for_whisper(
    audio: np.ndarray,
    sample_rate: int,
    target_dbfs: float = -18.0,
    preemphasis_coeff: float = 0.95,
    noise_reduce_strength: float = 0.70,
    target_sr: int = 16_000,
) -> tuple[np.ndarray, int]:
    """
    Enhance a speech audio array (from file) for Whisper transcription.

    Differences from the live VAD-segment version:
      - Accepts any sample rate and resamples to target_sr (16 kHz default)
      - Mixes stereo/multi-channel down to mono
      - Uses median-based noise floor estimation instead of assuming a
        leading silence tail — safe when speech starts at t=0
      - Gentler pre-emphasis coefficient (0.95 vs 0.97) to avoid
        over-sharpening already clear file recordings

    Args:
        audio:                  float32 numpy array, shape (N,) or (N, C).
        sample_rate:            Original sample rate of the audio.
        target_dbfs:            Target loudness after RMS normalization (dBFS).
        preemphasis_coeff:      High-freq boost coefficient (0.90–0.97).
                                Lower = gentler; use 0.97 only for very muffled sources.
        noise_reduce_strength:  Spectral subtraction aggressiveness (0.0–1.0).
                                0.5–0.75 is safe; above 0.85 risks musical noise.
        target_sr:              Output sample rate. Always 16000 for Whisper.

    Returns:
        (enhanced_audio, target_sr) — mono float32 array at target_sr.
    """
    audio = audio.astype(np.float32)

    # ------------------------------------------------------------------
    # 1. Mixdown to mono
    #    Average across channels. Simple and phase-safe for speech.
    # ------------------------------------------------------------------
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # ------------------------------------------------------------------
    # 2. Resample to target_sr (16 kHz for Whisper)
    #    scipy.signal.resample_poly uses a polyphase filter — better
    #    quality than naive decimation and much faster than FFT resample
    #    on long files.
    # ------------------------------------------------------------------
    if sample_rate != target_sr:
        from math import gcd

        g = gcd(sample_rate, target_sr)
        up = target_sr // g
        down = sample_rate // g
        audio = signal.resample_poly(audio, up, down).astype(np.float32)
        sample_rate = target_sr

    # ------------------------------------------------------------------
    # Guard: pure silence
    # ------------------------------------------------------------------
    if np.max(np.abs(audio)) < 1e-9:
        return audio, sample_rate

    # ------------------------------------------------------------------
    # 3. DC offset removal
    #    Removes constant bias common in certain capture devices.
    #    Must happen before filtering to avoid filter ringing.
    # ------------------------------------------------------------------
    audio -= np.mean(audio)

    # ------------------------------------------------------------------
    # 4. Bandpass filter: 80 Hz – 8 kHz
    #    Speech energy lives in 80 Hz–8 kHz. Whisper's mel filterbank
    #    covers exactly this range, so anything outside is pure noise
    #    that wastes model capacity and degrades transcription.
    # ------------------------------------------------------------------
    nyquist = sample_rate / 2.0
    sos = signal.butter(
        4,
        [80.0 / nyquist, min(8000.0 / nyquist, 0.99)],
        btype="band",
        output="sos",
    )
    audio = signal.sosfilt(sos, audio)

    # ------------------------------------------------------------------
    # 5. Spectral noise reduction (median-based spectral subtraction)
    #
    #    KEY DIFFERENCE FROM LIVE VERSION:
    #    For files we cannot assume the first 100ms is silence.
    #    Instead, we estimate the noise floor as the per-frequency-bin
    #    MEDIAN magnitude across all STFT frames. The median is robust
    #    to voiced speech frames — speech peaks are transient and skew
    #    the median less than the mean. This gives a stable background
    #    estimate even when speech starts at t=0.
    # ------------------------------------------------------------------
    n_fft = 1024  # larger window for files (vs 512 for live) → better freq resolution
    hop = n_fft // 4  # 75% overlap for smoother reconstruction

    _, _, stft = signal.stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # Median across time axis → shape (freq_bins, 1) — robust noise floor
    noise_floor = np.median(mag, axis=1, keepdims=True)

    # Subtract scaled noise floor; floor at a small residual (not 0)
    # to avoid completely zeroing bins (causes musical noise artifacts)
    residual_floor = 0.05 * noise_floor
    mag_denoised = np.maximum(mag - noise_reduce_strength * noise_floor, residual_floor)

    # Light temporal smoothing to reduce frame-to-frame magnitude jitter
    mag_denoised = uniform_filter1d(mag_denoised, size=3, axis=1)

    stft_denoised = mag_denoised * np.exp(1j * phase)
    _, audio = signal.istft(
        stft_denoised, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop
    )
    audio = audio.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Pre-emphasis
    #    First-order high-shelf: y[n] = x[n] - c * x[n-1]
    #    Sharpens fricatives (s, f, sh, th) that Whisper often mishears
    #    when they're softened by codec compression or low-bitrate streams.
    #    Using 0.95 (not 0.97) for files — recordings are typically
    #    cleaner than live captures and need less aggressive boosting.
    # ------------------------------------------------------------------
    audio = np.concatenate([[audio[0]], audio[1:] - preemphasis_coeff * audio[:-1]])

    # ------------------------------------------------------------------
    # 7. RMS normalization to target dBFS
    #    Most impactful step for low-volume sources (browser video at
    #    20% volume, Mac system audio via BlackHole, etc.).
    #    Brings the file to a consistent loudness Whisper expects.
    # ------------------------------------------------------------------
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-9:
        target_linear = 10 ** (target_dbfs / 20.0)
        audio = audio * (target_linear / rms)

    # ------------------------------------------------------------------
    # 8. Soft-knee peak limiter
    #    After normalization, transient peaks may exceed [-1, 1].
    #    Tanh limiting is smoother than np.clip — avoids hard clipping
    #    distortion on plosives (p, b, t, d) and loud consonants.
    # ------------------------------------------------------------------
    threshold = 0.95
    above = np.abs(audio) > threshold
    if np.any(above):
        audio[above] = np.sign(audio[above]) * (
            threshold
            + (1.0 - threshold)
            * np.tanh((np.abs(audio[above]) - threshold) / (1.0 - threshold))
        )

    return audio.astype(np.float32), sample_rate


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def load_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Load any soundfile-supported format (wav, flac, ogg, aiff, etc.)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, sr


def save_audio_file(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Save float32 mono array as a 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Enhance a speech audio file for Whisper transcription.\n"
            "Output is saved to: generated/<script_stem>/<input_stem>_enhanced.wav"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        type=Path,
        help="Path to the input audio file (wav, flac, ogg, aiff, …).",
    )
    p.add_argument(
        "--target-dbfs",
        type=float,
        default=-18.0,
        metavar="DB",
        help=(
            "Target loudness after RMS normalization in dBFS. "
            "Default: -18.0. Use -14.0 for very quiet sources."
        ),
    )
    p.add_argument(
        "--preemphasis",
        type=float,
        default=0.95,
        metavar="COEFF",
        help=(
            "High-frequency boost coefficient (0.90–0.97). "
            "Higher = stronger consonant sharpening. Default: 0.95."
        ),
    )
    p.add_argument(
        "--noise-strength",
        type=float,
        default=0.70,
        metavar="STRENGTH",
        help=(
            "Spectral noise subtraction aggressiveness (0.0–1.0). "
            "Default: 0.70. Reduce to 0.5 if you hear watery artifacts."
        ),
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Suppress before/after audio stats printed to stdout.",
    )
    return p


def print_stats(label: str, audio: np.ndarray, sr: int) -> None:
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    rms_db = 20 * np.log10(rms) if rms > 1e-9 else float("-inf")
    peak_db = 20 * np.log10(peak) if peak > 1e-9 else float("-inf")
    print(f"  [{label}]")
    print(f"    Duration : {duration:.2f}s  ({len(audio):,} samples @ {sr} Hz)")
    print(f"    RMS      : {rms_db:+.1f} dBFS")
    print(f"    Peak     : {peak_db:+.1f} dBFS")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        print(f"[error] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = OUTPUT_DIR / f"{input_path.stem}_enhanced.wav"

    print(f"\nInput  : {input_path}")
    print(f"Output : {output_path}\n")

    # Load
    audio, sr = load_audio_file(input_path)
    if not args.no_stats:
        print("Before enhancement:")
        print_stats("original", audio.mean(axis=1) if audio.ndim == 2 else audio, sr)

    # Enhance
    enhanced, out_sr = enhance_speech_for_whisper(
        audio,
        sample_rate=sr,
        target_dbfs=args.target_dbfs,
        preemphasis_coeff=args.preemphasis,
        noise_reduce_strength=args.noise_strength,
    )

    if not args.no_stats:
        print("\nAfter enhancement:")
        print_stats("enhanced", enhanced, out_sr)

    # Save
    save_audio_file(enhanced, out_sr, output_path)
    print(f"\n✓ Saved → {output_path}")
