import datetime
import asyncio
import os
from typing import Optional
import threading
import sys
from gtts import gTTS
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
console = Console()


class AdvancedTTSEngine:
    def __init__(self, rate: int = 200, voice_map: dict = None, output_dir: str = None):
        self.rate = rate
        self.voice_map = voice_map or {
            "Emma": "female_voice_id", "Liam": "male_voice_id"}
        self.lock = threading.Lock()
        pygame.mixer.init()
        self.temp_files = []  # Track temporary audio files
        self.output_dir = output_dir if output_dir else script_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache = {}  # In-memory cache for TTS audio
        self.channel = pygame.mixer.Channel(
            0)  # Dedicated channel for playback

    def _get_audio_filename(self, speaker_name: str, text: str) -> str:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_text = ''.join(c for c in text[:30] if c.isalnum() or c in (
            ' ', '_')).strip().replace(' ', '_')
        filename = os.path.join(
            self.output_dir, f"tts_{speaker_name}_{timestamp}_{safe_text}.mp3")
        self.temp_files.append(filename)
        return filename

    def speak(self, text: str, speaker_name: str = "Agent"):
        with self.lock:
            # Skip empty or whitespace-only text
            if not text.strip():
                logger.warning(f"Skipping empty TTS text for {speaker_name}")
                return
            # Check cache for existing audio
            cache_key = f"{speaker_name}:{text}"
            if cache_key in self.cache:
                file_path = self.cache[cache_key]
                logger.info(f"Using cached audio: {file_path}")
            else:
                file_path = self._get_audio_filename(speaker_name, text)
                try:
                    tts = gTTS(text=text, lang='en')
                    tts.save(file_path)
                    self.cache[cache_key] = file_path
                    logger.success(f"TTS audio saved: {file_path}")
                except Exception as e:
                    logger.error(f"Error in TTS generation: {e}")
                    raise

            try:
                self.channel.play(pygame.mixer.Sound(
                    file_path))  # Non-blocking playback
            except Exception as e:
                logger.error(f"Error in audio playback: {e}")
                raise

    async def speak_async(self, text: str, speaker_name: str = "Agent"):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text, speaker_name)
        await asyncio.sleep(0.01)  # Minimal delay to yield control

    def cleanup(self):
        """Remove temporary audio files and clear cache."""
        with self.lock:
            for file_path in self.temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting temp file {file_path}: {e}")
            self.temp_files.clear()
            self.cache.clear()


class Agent:
    def __init__(self, name: Optional[str], system_prompt: str, model: str = "llama3.1", session_id: str = "", output_dir: str = None) -> None:
        if name is None:
            raise ValueError("Agent name must be provided or set in subclass")
        self.name: str = name
        self.ollama: Ollama = Ollama(
            model=model, system=system_prompt, session_id=session_id, temperature=0.3)
        self.chat_history = self.ollama.chat_history
        self.tts = AdvancedTTSEngine(
            rate=200 if name == "Emma" else 180, output_dir=output_dir)

    async def generate_response(self, external_message: str) -> str:
        content = ""
        chunk_buffer = ""  # Accumulate chunks until boundary
        async for chunk in self.ollama.stream_chat(query=external_message):
            content += chunk
            logger.debug(f"Received chunk: '{chunk}'")

            # Append chunk to buffer
            chunk_buffer += chunk

            # Check if buffer ends with space or newline
            if chunk_buffer.endswith((' ', '\n')) and chunk_buffer.strip():
                # Speak the current batch of chunks
                await self.tts.speak_async(f"{self.name}: {chunk_buffer}", speaker_name=self.name)
                logger.info(
                    f"Spoke buffer ending with space/newline: '{chunk_buffer}'")
                chunk_buffer = ""  # Reset buffer
            # Check if chunk starts with space or newline
            elif chunk.startswith((' ', '\n')) and chunk_buffer.strip():
                # Speak the previous batch of chunks (before appending current chunk)
                prev_buffer = chunk_buffer[:-len(chunk)]
                if prev_buffer.strip():
                    await self.tts.speak_async(f"{self.name}: {prev_buffer}", speaker_name=self.name)
                    logger.info(
                        f"Spoke buffer before space/newline chunk: '{prev_buffer}'")
                chunk_buffer = chunk  # Start new buffer with current chunk

        # Speak any remaining buffered chunks
        if chunk_buffer.strip():
            await self.tts.speak_async(f"{self.name}: {chunk_buffer}", speaker_name=self.name)
            logger.info(f"Spoke final buffered chunks: '{chunk_buffer}'")

        return content

    def clear_history(self) -> None:
        self.chat_history.clear()

    def cleanup(self):
        """Cleanup TTS resources."""
        self.tts.cleanup()


class Interviewer(Agent):
    def __init__(self, model: str = "llama3.1", name: Optional[str] = None, output_dir: str = None, **kwargs) -> None:
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
        super().__init__(name=name, system_prompt=system_prompt,
                         model=model, output_dir=output_dir, **kwargs)


class Applicant(Agent):
    def __init__(self, model: str = "llama3.1", name: Optional[str] = None, output_dir: str = None, **kwargs) -> None:
        name = name or "Liam"
        system_prompt = (
            f"You are {name}, a job applicant applying for a software engineering position. "
            "You have a strong background in Python, Java, and web development, with 3 years of experience. "
            f"Respond to the interviewer's questions professionally, concisely, and with relevant details. "
            "If asked about weaknesses, be honest but frame them positively."
        )
        super().__init__(name=name, system_prompt=system_prompt,
                         model=model, output_dir=output_dir, **kwargs)


async def main():
    output_dir = os.path.join(script_dir, "generated", "audio_output")
    interviewer = Interviewer(output_dir=output_dir)
    applicant = Applicant(output_dir=output_dir)
    playback_overlap = 0.5  # Seconds to overlap playback for natural flow

    try:
        # Start the interview
        question_task = asyncio.create_task(
            interviewer.generate_response("Start the interview."))
        question = await question_task

        for _ in range(3):
            # Start applicant's response while interviewer's audio is still playing
            await asyncio.sleep(playback_overlap)  # Allow slight overlap
            applicant_task = asyncio.create_task(
                applicant.generate_response(question))
            response = await applicant_task

            # Start interviewer's next question with overlap
            await asyncio.sleep(playback_overlap)
            question_task = asyncio.create_task(
                interviewer.generate_response(response))
            question = await question_task
    finally:
        interviewer.cleanup()
        applicant.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
