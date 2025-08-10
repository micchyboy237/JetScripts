import librosa
import numpy as np
from faster_whisper import WhisperModel
from typing import Iterator, Tuple, Optional
from pathlib import Path


class AudioTranscriber:
    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        """Initialize the transcriber with a specified model size and device."""
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type)
        self.sample_rate = 16000  # Whisper expects 16kHz audio

    def load_audio(self, audio_path: str | Path) -> np.ndarray:
        """Load audio file using librosa."""
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio

    def split_audio(self, audio: np.ndarray, segment_duration: float = 5.0, overlap_duration: float = 0.5) -> Iterator[Tuple[np.ndarray, float, float]]:
        """Split audio into segments with optional overlap to prevent information loss.

        Args:
            audio: Input audio array.
            segment_duration: Duration of each segment in seconds.
            overlap_duration: Duration of overlap between segments in seconds.

        Yields:
            Tuple containing the audio segment, start time, and end time.
        """
        samples_per_segment = int(self.sample_rate * segment_duration)
        overlap_samples = int(self.sample_rate * overlap_duration)
        step_samples = samples_per_segment - overlap_samples  # Step size adjusted for overlap
        total_samples = len(audio)

        if step_samples <= 0:
            raise ValueError("Overlap duration must be less than segment duration.")

        for start_sample in range(0, total_samples, step_samples):
            end_sample = min(start_sample + samples_per_segment, total_samples)
            segment = audio[start_sample:end_sample]
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            yield segment, start_time, end_time

    def transcribe_segment(self, segment: np.ndarray, start_time: float, end_time: float, language: Optional[str] = None) -> Tuple[str, float, float]:
        """Transcribe a single audio segment."""
        segments, _ = self.model.transcribe(
            segment,
            beam_size=1,  # Optimize for speed
            temperature=0,  # Deterministic output
            language=language,
            without_timestamps=False
        )
        text = " ".join(seg.text for seg in segments)
        return text.strip(), start_time, end_time

    def transcribe_audio(self, audio_path: str | Path, segment_duration: float = 5.0, overlap_duration: float = 0.5, language: Optional[str] = None) -> Iterator[Tuple[str, float, float]]:
        """Transcribe audio file in segments with overlap and yield text with timestamps.

        Args:
            audio_path: Path to the audio file.
            segment_duration: Duration of each segment in seconds.
            overlap_duration: Duration of overlap between segments in seconds.
            language: Language code for transcription (None for auto-detection).

        Yields:
            Tuple of transcribed text, start time, and end time for each segment.
        """
        audio = self.load_audio(audio_path)
        for segment, start_time, end_time in self.split_audio(audio, segment_duration, overlap_duration):
            if len(segment) > 0:  # Skip empty segments
                text, start, end = self.transcribe_segment(
                    segment, start_time, end_time, language)
                if text:  # Only yield non-empty transcriptions
                    yield text, start, end


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Path to the downloaded audio file
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_stream_mic/original_stream.wav"
    audio_file = Path(audio_file)
    model_size = "small"  # Using small model
    device = "cpu"  # Default to CPU for compatibility with Mac M1
    segment_duration = 5.0  # 5-second segments
    overlap_duration = 1.0  # 1.0-second overlap to prevent information loss
    language = "en"  # Specify language (optional, set to English)

    try:
        # Initialize the transcriber
        transcriber = AudioTranscriber(
            model_size=model_size,
            device=device,
            compute_type="int8"  # Optimized for CPU
        )

        # Verify the audio file exists
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file '{audio_file}' not found.")

        # Transcribe the audio file
        print(f"Transcribing {audio_file} with {overlap_duration}s overlap...")
        for text, start_time, end_time in transcriber.transcribe_audio(
            audio_path=audio_file,
            segment_duration=segment_duration,
            overlap_duration=overlap_duration,
            language=language
        ):
            # Format and print the transcription with timestamps
            print(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        sys.exit(1)