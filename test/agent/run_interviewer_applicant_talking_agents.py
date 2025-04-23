import datetime
import asyncio
import os
from typing import Optional
import threading
import sys
import time
from jet.data.utils import generate_unique_hash
from jet.llm.ollama.base import Ollama
from jet.logger.logger import CustomLogger
import pygame
from rich.console import Console
from rich.text import Text

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
console = Console()  # Rich console for enhanced text display


class AdvancedTTSEngine:
    def __init__(self, rate: int = 200, voice_map: dict = None):
        self.rate = rate
        # Map agent names to voice IDs
        self.voice_map = voice_map or {
            "Emma": "female_voice_id", "Liam": "male_voice_id"}
        self.lock = threading.Lock()
        pygame.mixer.init()
        self.temp_files = []  # Track temporary audio files

    def _get_audio_filename(self, speaker_name: str, text: str) -> str:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_text = ''.join(c for c in text[:30] if c.isalnum() or c in (
            ' ', '_')).strip().replace(' ', '_')
        filename = os.path.join(
            script_dir, f"tts_{speaker_name}_{timestamp}_{safe_text}.mp3")
        self.temp_files.append(filename)
        return filename

    def speak(self, text: str, speaker_name: str = "Agent"):
        with self.lock:
            file_path = self._get_audio_filename(speaker_name, text)
            try:
                # Placeholder for advanced TTS (e.g., ElevenLabs API)
                # Replace with actual API call, e.g., elevenlabs.generate(text, voice=self.voice_map.get(speaker_name, "default"))
                # Fallback to gTTS for demonstration
                tts = gTTS(text=text, lang='en')
                tts.save(file_path)

                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                logger.success(f"TTS audio saved: {file_path}")
            except Exception as e:
                logger.error(f"Error in TTS generation/playback: {e}")
                raise

    async def speak_async(self, text: str, speaker_name: str = "Agent"):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text, speaker_name)
        await asyncio.sleep(0.05)  # Reduced delay for smoother transitions

    def cleanup(self):
        """Remove temporary audio files."""
        with self.lock:
            for file_path in self.temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting temp file {file_path}: {e}")
            self.temp_files.clear()


class Agent:
    def __init__(self, name: Optional[str], system_prompt: str, model: str = "llama3.1", session_id: str = "") -> None:
        if name is None:
            raise ValueError("Agent name must be provided or set in subclass")
        self.name: str = name
        self.ollama: Ollama = Ollama(
            model=model, system=system_prompt, session_id=session_id, temperature=0.3)
        self.chat_history = self.ollama.chat_history
        self.tts = AdvancedTTSEngine(rate=200 if name == "Emma" else 180)

    async def generate_response(self, external_message: str) -> str:
        content = ""
        # Display speaker name
        console.print(f"[bold blue]{self.name}:[/bold blue] ", end="")
        async for chunk in self.ollama.stream_chat(query=external_message):
            content += chunk
            # Display chunk in real-time with rich formatting
            console.print(Text(chunk, style="green"), end="", soft_wrap=True)
            sys.stdout.flush()
        console.print()  # Newline after response
        # Speak the full response with speaker prefix
        await self.tts.speak_async(f"{self.name}: {content}", speaker_name=self.name)
        return content

    def clear_history(self) -> None:
        self.chat_history.clear()

    def cleanup(self):
        """Cleanup TTS resources."""
        self.tts.cleanup()


class Interviewer(Agent):
    def __init__(self, model: str = "llama3.1", name: Optional[str] = None, **kwargs) -> None:
        name = name or "Emma"
        system_prompt = (
            f"You are {name}, a professional job interviewer for a software engineering position. "
            "Follow a structured agenda to assess the candidate's technical skills, experience, "
            "and problem-solving abilities. Ask one clear, relevant question at a time, following this agenda in order:\n"
            "1. Introduction: Ask the candidate to tell you about themselves and their background.\n"
            "2. Technical Skills: Ask about their experience with Python and how they've used it in a project.\n"
            "3. Problem-Solving: Ask about a challenging technical problem they faced and how they solved it.\n"
            "4. Teamwork: Ask about a time they worked in a team to complete a software project and their role.\n"
            "5. Weaknesses: Ask about a professional weakness they've identified and how they are working to improve it.\n"
            "6. Closing: Ask if they have any questions about the role or the company.\n"
            "Use the chat history to determine which agenda topic to address next, ensuring you progress through the topics in order.\n"
            "\n"
            f"Incorporate the candidate's previous response to maintain a natural conversation flow (e.g., 'Thank you for sharing that, Liam. [Next question]').\n"
            f"Be polite, professional, and concise. After asking the final agenda topic (Closing), wait for the candidate's response.\n"
            "If the candidate raises questions or concerns, address them appropriately and then ask again if they have any further questions or concerns, repeating this process until they have no more.\n"
            "If the candidate indicates they have no further questions or concerns (e.g., 'No questions' or 'I'm good'), end the interview politely and include '[TERMINATE]' in your final message."
        )
        super().__init__(name=name, system_prompt=system_prompt, model=model, **kwargs)


class Applicant(Agent):
    def __init__(self, model: str = "llama3.1", name: Optional[str] = None, **kwargs) -> None:
        name = name or "Liam"
        system_prompt = (
            f"You are {name}, a job applicant applying for a software engineering position. "
            "You have a strong background in Python, Java, and web development, with 3 years of experience. "
            f"Respond to the interviewer's questions professionally, concisely, and with relevant details. "
            "If asked about weaknesses, be honest but frame them positively."
        )
        super().__init__(name=name, system_prompt=system_prompt, model=model, **kwargs)


# Example usage (for testing)
async def main():
    interviewer = Interviewer()
    applicant = Applicant()
    try:
        # Simulate a few interview rounds
        question = await interviewer.generate_response("Start the interview.")
        for _ in range(3):  # Limit to 3 exchanges for brevity
            response = await applicant.generate_response(question)
            question = await interviewer.generate_response(response)
    finally:
        # Cleanup resources
        interviewer.cleanup()
        applicant.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
