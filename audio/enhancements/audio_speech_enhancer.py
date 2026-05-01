import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def enhance_speech_segment(
    segment: np.ndarray,
    sample_rate: int = 16000,
    target_dbfs: float = -18.0,
    preemphasis_coeff: float = 0.97,
    noise_reduce_strength: float = 0.75,
) -> np.ndarray:
    """
    Enhance a full VAD-accumulated speech segment for Whisper transcription.

    Applies a chain of signal processing steps tuned for speech clarity:
      1. DC offset removal
      2. Bandpass filter (80 Hz – 8 kHz) — isolates speech frequencies
      3. Spectral noise reduction — attenuates stationary background noise
      4. Pre-emphasis — boosts high-freq consonants (s, t, f, sh)
      5. RMS normalization — brings low-volume audio to a consistent level
      6. Peak limiting — prevents clipping distortion after gain

    Args:
        segment:              float32 numpy array, mono, range [-1.0, 1.0]
                              This is a FULL utterance (1–30s), NOT a 32ms VAD frame.
        sample_rate:          Should match Whisper's expected rate (default 16000 Hz).
        target_dbfs:          Target loudness after normalization in dBFS.
                              -18.0 is a good balance — loud enough, not clipped.
        preemphasis_coeff:    High-frequency boost coefficient (0.95–0.99).
                              Higher = stronger consonant emphasis.
        noise_reduce_strength: How aggressively to suppress noise floor (0.0–1.0).
                              Higher = more suppression, risk of artifacts on soft speech.

    Returns:
        Enhanced float32 numpy array, same length as input, range [-1.0, 1.0].
    """
    audio = segment.astype(np.float32)

    # --- Guard: return silence as-is ---
    if np.max(np.abs(audio)) < 1e-9:
        return audio

    # 1. DC offset removal
    #    Removes any constant bias in the waveform that shifts it away from zero.
    #    Common with certain capture devices (e.g. BlackHole, some USB mics).
    #    A DC offset skews RMS calculations and distorts mel spectrograms.
    audio -= np.mean(audio)

    # 2. Bandpass filter: 80 Hz – 8 kHz
    #    - Below 80 Hz: rumble, HVAC hum, desk vibration — not speech
    #    - Above 8 kHz: hiss, ultrasonic artifacts — Whisper's mel filterbank
    #      goes up to 8 kHz anyway, so anything above is irrelevant noise
    nyquist = sample_rate / 2.0
    low_cut = 80.0 / nyquist
    high_cut = min(8000.0 / nyquist, 0.99)  # clamp below Nyquist
    sos = signal.butter(4, [low_cut, high_cut], btype="band", output="sos")
    audio = signal.sosfilt(sos, audio)

    # 3. Spectral noise reduction (simple spectral subtraction)
    #    Estimates the noise floor from the first 0.1s of the segment
    #    (assumed to be the tail of silence before speech started).
    #    Then subtracts a scaled version of that noise spectrum from all frames.
    #    This is the most impactful step for system audio / BlackHole captures
    #    which often have a constant background hiss or hum.
    noise_sample_len = int(0.1 * sample_rate)  # 100ms noise reference
    if len(audio) > noise_sample_len * 3:
        noise_ref = audio[:noise_sample_len]
        n_fft = 512
        hop = n_fft // 2

        # Estimate noise magnitude spectrum from reference region
        _, _, noise_stft = signal.stft(
            noise_ref, fs=sample_rate, nperseg=n_fft, noverlap=hop
        )
        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

        # STFT of full segment
        freqs, times, stft = signal.stft(
            audio, fs=sample_rate, nperseg=n_fft, noverlap=hop
        )
        mag = np.abs(stft)
        phase = np.angle(stft)

        # Subtract noise floor, floor at 0 to avoid negative magnitudes
        mag_denoised = np.maximum(mag - noise_reduce_strength * noise_mag, 0.0)

        # Smooth the magnitude to reduce musical noise artifacts
        mag_denoised = uniform_filter1d(mag_denoised, size=3, axis=1)

        # Reconstruct signal
        stft_denoised = mag_denoised * np.exp(1j * phase)
        _, audio = signal.istft(
            stft_denoised, fs=sample_rate, nperseg=n_fft, noverlap=hop
        )
        audio = audio.astype(np.float32)

        # istft may produce slightly different length — trim/pad to match original
        if len(audio) > len(segment):
            audio = audio[: len(segment)]
        elif len(audio) < len(segment):
            audio = np.pad(audio, (0, len(segment) - len(audio)))

    # 4. Pre-emphasis filter
    #    Applies a first-order high-shelf boost: y[n] = x[n] - coeff * x[n-1]
    #    Compensates for the natural high-frequency rolloff in human speech.
    #    Sharpens fricatives (s, f, sh, th) which Whisper commonly mishears
    #    when they're muffled by low-quality captures or codec compression.
    audio = np.concatenate([[audio[0]], audio[1:] - preemphasis_coeff * audio[:-1]])

    # 5. RMS normalization to target dBFS
    #    This is the most critical step for the low-volume problem:
    #    browser video player at 20% volume, Mac system audio, etc.
    #    Brings the segment to a consistent loudness regardless of source gain.
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-9:
        target_linear = 10 ** (target_dbfs / 20.0)
        audio = audio * (target_linear / rms)

    # 6. Peak limiter — soft knee to prevent hard clipping
    #    After normalization, occasional peaks may exceed [-1, 1].
    #    Hard clip (np.clip) creates audible distortion on transients.
    #    This applies a smooth tanh-based limiter instead.
    threshold = 0.95
    above = np.abs(audio) > threshold
    audio[above] = np.sign(audio[above]) * (
        threshold
        + (1.0 - threshold)
        * np.tanh((np.abs(audio[above]) - threshold) / (1.0 - threshold))
    )

    return audio.astype(np.float32)
