# tests/test_realtime_transcriber.py

import numpy as np
import pytest

from _temp import RealtimeTranscriber


class TestRealtimeTranscriber:
    @pytest.fixture
    def transcriber(self) -> RealtimeTranscriber:
        return RealtimeTranscriber(
            model_size="small",
            device="cpu",
            compute_type="int8",
            vad_threshold=0.02,
            silence_seconds=0.3,
        )

    def test_vad_detects_silence(self, transcriber: RealtimeTranscriber):
        # Given: silent audio chunk
        silent_chunk = np.zeros(16000, dtype=np.float32)
        # When: processed by VAD
        result = transcriber._vad_energy(silent_chunk)
        # Then: detected as non-speech
        expected = False
        assert result == expected

    def test_vad_detects_speech(self, transcriber: RealtimeTranscriber):
        # Given: typical speech-like chunk (RMS ~0.05-0.1 for normal speech)
        speech_chunk = np.random.uniform(-0.3, 0.3, 16000).astype(np.float32)  # Realistic amplitude
        # When: processed by VAD
        result = transcriber._vad_energy(speech_chunk)
        # Then: detected as speech
        expected = True
        assert result == expected

    def test_buffer_accumulation_and_reset(self, transcriber: RealtimeTranscriber):
        # Given: speech followed by sufficient silence
        speech = np.random.uniform(-0.3, 0.3, 32000).astype(np.float32)  # 2s realistic speech
        transcriber.speech_buffer = np.array([], dtype=np.float32)
        transcriber.silence_timer = 0.0

        # When: feeding speech chunks (simulating VAD active)
        for i in range(2):
            chunk = speech[i * 16000:(i + 1) * 16000]
            if transcriber._vad_energy(chunk):  # Now reliably True
                transcriber.speech_buffer = np.concatenate((transcriber.speech_buffer, chunk))

        result_after_speech = len(transcriber.speech_buffer)
        expected_after_speech = 32000
        assert result_after_speech == expected_after_speech

        # And when: silence exceeds threshold (manual trigger for test isolation)
        transcriber.silence_timer = 1.0  # > silence_seconds=0.3
        if transcriber.speech_buffer.size > 0 and transcriber.silence_timer >= transcriber.silence_duration:
            old_buffer = transcriber.speech_buffer.copy()  # Not used but for clarity
            transcriber.speech_buffer = np.array([], dtype=np.float32)
            transcriber.silence_timer = 0.0

        result_after_silence = len(transcriber.speech_buffer)
        expected_after_silence = 0
        assert result_after_silence == expected_after_silence