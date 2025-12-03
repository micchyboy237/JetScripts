from typing import List, TypedDict, Callable
import torch
from silero_vad.utils_vad import get_speech_timestamps


class SpeechSegment(TypedDict):
    start: int | float      # samples or seconds
    end: int | float        # samples or seconds
    prob: float             # average speech probability in segment
    duration: float         # duration in seconds


@torch.no_grad()
def get_speech_timestamps_enhanced(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 1,
    progress_tracking_callback: Callable[[float], None] | None = None,
) -> List[SpeechSegment]:
    """
    Enhanced get_speech_timestamps that includes:
    - Average speech probability per detected segment
    - Duration in seconds
    - Fully typed, clean, no unused variables
    """
    if sampling_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports only 8000 or 16000 Hz (or multiples)")

    window_size_samples = 512 if sampling_rate == 16000 else 256
    model.reset_states()

    # === 1. Collect per-window speech probabilities ===
    speech_probs: List[float] = []
    num_windows = (len(audio) + window_size_samples - 1) // window_size_samples

    for i in range(num_windows):
        start = i * window_size_samples
        end = min(start + window_size_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))

        prob = model(chunk.unsqueeze(0), sampling_rate).item()
        speech_probs.append(prob)

        if progress_tracking_callback:
            progress = min((i + 1) / num_windows * 100, 100.0)
            progress_tracking_callback(progress)

    # === 2. Get reliable speech segments using original logic ===
    segments = get_speech_timestamps(
        audio=audio,
        model=model,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,  # work in samples internally
    )

    # === 3. Map probabilities to each segment ===
    enhanced: List[SpeechSegment] = []
    for seg in segments:
        start_sample = seg["start"]
        end_sample = seg["end"]

        # Find window indices that overlap with this speech segment
        start_idx = start_sample // window_size_samples
        end_idx = (end_sample + window_size_samples - 1) // window_size_samples

        start_idx = max(0, start_idx)
        end_idx = min(len(speech_probs), end_idx)

        window_probs = speech_probs[start_idx:end_idx]
        avg_prob = sum(window_probs) / len(window_probs) if window_probs else 0.0
        duration_sec = (end_sample - start_sample) / sampling_rate

        enhanced.append(SpeechSegment(
            start=round(start_sample / sampling_rate, time_resolution) if return_seconds else start_sample,
            end=round(end_sample / sampling_rate, time_resolution) if return_seconds else end_sample,
            prob=round(avg_prob, 4),
            duration=round(duration_sec, 3),
        ))

    return enhanced

if __name__ == "__main__":
    from silero_vad.utils_vad import read_audio
    import torch

    # Load model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    # or use ONNX: OnnxWrapper(...)

    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    audio = read_audio(audio_file, sampling_rate=16000)

    segments = get_speech_timestamps_enhanced(
        audio,
        model,
        threshold=0.5,
        sampling_rate=16000,
        return_seconds=True,
        progress_tracking_callback=print  # optional
    )

    for seg in segments:
        print(f"[{seg['start']:.2f} - {seg['end']:.2f}] duration={seg['duration']}s prob={seg['prob']:.3f}")