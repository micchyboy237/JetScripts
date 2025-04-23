import datetime
import asyncio
import os
import shutil
from typing import Optional, List
import threading
import sys
from gtts import gTTS
from jet.wordnet.sentence import split_sentences
from pydub import AudioSegment
from jet.data.utils import generate_unique_hash
from jet.llm.ollama.base import Ollama
from jet.logger.logger import CustomLogger
import pygame
from rich.console import Console
from rich.text import Text
import time

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
        self.combined_files = []  # Track combined audio files
        self.output_dir = output_dir if output_dir else script_dir
        # Reset output directory on initialization
        self._reset_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache = {}  # In-memory cache for TTS audio
        self.channel = pygame.mixer.Channel(
            0)  # Dedicated channel for playback

    def _reset_output_dir(self):
        """Reset the output directory by removing all files."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            logger.info(f"Cleared output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_audio_filename(self, speaker_name: str, text: str, prefix: str = "tts") -> str:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_text = ''.join(c for c in text[:30] if c.isalnum() or c in (
            ' ', '_')).strip().replace(' ', '_')
        filename = os.path.join(
            self.output_dir, f"{prefix}_{speaker_name}_{timestamp}_{safe_text}.mp3")
        if prefix == "tts":
            self.temp_files.append(filename)
        else:
            self.combined_files.append(filename)
        return filename

    def speak(self, text: str, speaker_name: str = "Agent") -> Optional[str]:
        with self.lock:
            # Skip empty or whitespace-only text
            if not text.strip():
                logger.warning(f"Skipping empty TTS text for {speaker_name}")
                return None

            # Create cache key
            cache_key = f"{speaker_name}:{text}"

            if cache_key in self.cache:
                # Using cached audio
                file_path = self.cache[cache_key]
            else:
                file_path = self._get_audio_filename(speaker_name, text)
                try:
                    # Create new TTS audio file
                    tts = gTTS(text=text, lang='en')
                    tts.save(file_path)
                    self.cache[cache_key] = file_path
                except Exception as e:
                    logger.error(f"Error in TTS generation: {e}")
                    raise

            try:
                # Non-blocking playback
                self.channel.play(pygame.mixer.Sound(file_path))
                return file_path
            except Exception as e:
                logger.error(f"Error in audio playback: {e}")
                raise

    async def speak_async(self, text: str, speaker_name: str = "Agent") -> Optional[str]:
        loop = asyncio.get_event_loop()
        file_path = await loop.run_in_executor(None, self.speak, text, speaker_name)
        await asyncio.sleep(0.01)
        return file_path

    def combine_audio_files(self, file_paths: List[str], speaker_name: str, text: str) -> Optional[str]:
        if not file_paths:
            logger.warning("No audio files to combine")
            return None
        with self.lock:
            start_time = time.time()
            try:
                combined = AudioSegment.empty()
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        audio = AudioSegment.from_mp3(file_path)
                        combined += audio
                    else:
                        logger.warning(f"Audio file not found: {file_path}")
                if not combined:
                    logger.warning("No valid audio segments to combine")
                    return None
                output_file = self._get_audio_filename(
                    speaker_name, text, prefix="combined")
                combined.export(output_file, format="mp3", bitrate="64k")
                logger.success(f"Combined audio saved: {output_file}")
                logger.info(
                    f"Combining audio took {time.time() - start_time:.2f} seconds")

                # Clean up fragment files after successful combination
                self._cleanup_temp_files()

                return output_file
            except Exception as e:
                logger.error(f"Error combining audio files: {e}")
                logger.info(
                    f"Combining audio failed after {time.time() - start_time:.2f} seconds")
                return None

    async def combine_audio_files_async(self, file_paths: List[str], speaker_name: str, text: str) -> Optional[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.combine_audio_files, file_paths, speaker_name, text)

    def _cleanup_temp_files(self):
        """Remove only temporary audio files, preserving combined files."""
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

    def cleanup(self):
        """Remove temporary audio files and clear cache, preserving combined files."""
        self._cleanup_temp_files()


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
        buffer = ""
        audio_files = []

        async for chunk in self.ollama.stream_chat(query=external_message):
            content += chunk
            buffer += chunk

            sentences = split_sentences(buffer)

            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    clean_sentence = sentence.strip()
                    if clean_sentence:
                        # Speak buffered sentence
                        file_path = await self.tts.speak_async(f"{self.name}: {clean_sentence}", speaker_name=self.name)
                        if file_path:
                            audio_files.append(file_path)
                buffer = sentences[-1]

        final_sentence = buffer.strip()
        if final_sentence:
            # Speak final buffered sentence
            file_path = await self.tts.speak_async(f"{self.name}: {final_sentence}", speaker_name=self.name)
            if file_path:
                audio_files.append(file_path)

        if audio_files and len(audio_files) > 1:
            asyncio.create_task(self.tts.combine_audio_files_async(
                audio_files, self.name, content))
            logger.info("Scheduled background audio combining")
        elif audio_files:
            logger.info(
                f"Single audio file, skipping combine: {audio_files[0]}")

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
    playback_overlap = 0.5

    try:
        question_task = asyncio.create_task(
            interviewer.generate_response("Start the interview."))
        question = await question_task

        for _ in range(3):
            await asyncio.sleep(playback_overlap)
            applicant_task = asyncio.create_task(
                applicant.generate_response(question))
            response = await applicant_task
            await asyncio.sleep(playback_overlap)
            question_task = asyncio.create_task(
                interviewer.generate_response(response))
            question = await question_task
    finally:
        interviewer.cleanup()
        applicant.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
