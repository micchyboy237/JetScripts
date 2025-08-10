import logging
from typing import Optional
import speech_recognition as sr

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """Handles voice recognition and transcription using speech_recognition."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def adjust_for_ambient_noise(self, duration: float = 1.0) -> None:
        """Adjusts recognizer for ambient noise."""
        try:
            logger.info("Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=duration)
            logger.info("Ambient noise adjustment completed.")
        except Exception as e:
            logger.error(f"Failed to adjust for ambient noise: {e}")
            raise

    def listen(self, timeout: int = 5, phrase_time_limit: Optional[int] = None) -> Optional[sr.AudioData]:
        """Captures audio from the microphone."""
        try:
            logger.info("Listening for audio input...")
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logger.info("Audio captured successfully.")
            return audio
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out, no speech detected.")
            return None
        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
            raise

    def transcribe(self, audio: Optional[sr.AudioData]) -> Optional[str]:
        """Transcribes audio to text using Google's Speech Recognition API."""
        if audio is None:
            logger.warning("No audio provided for transcription.")
            return None
        try:
            logger.info("Transcribing audio...")
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Transcription successful: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand the audio.")
            return None
        except sr.RequestError as e:
            logger.error(f"Transcription API error: {e}")
            return None


def transcribe_speech() -> Optional[str]:
    """Main function to transcribe speech from microphone."""
    transcriber = SpeechTranscriber()
    transcriber.adjust_for_ambient_noise()
    audio = transcriber.listen()
    return transcriber.transcribe(audio)


if __name__ == "__main__":
    result = transcribe_speech()
    if result:
        print(f"Transcribed text: {result}")
    else:
        print("No transcription available.")
