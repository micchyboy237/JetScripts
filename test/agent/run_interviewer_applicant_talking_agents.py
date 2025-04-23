import datetime
import asyncio
import os
from typing import Optional
from jet.data.utils import generate_unique_hash
from jet.llm.ollama.base import Ollama
from jet.logger.logger import CustomLogger
from gtts import gTTS
import pygame
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)


class TTSEngine:
    def __init__(self, rate: int = 200):
        self.rate = rate  # Kept for compatibility; gTTS doesn't directly support rate
        self.lock = threading.Lock()  # Ensure thread-safe audio generation and playback
        pygame.mixer.init()  # Initialize pygame mixer for audio playback

    def _get_audio_filename(self, speaker_name: str, text: str) -> str:
        """Generate a unique filename using timestamp and a hash."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_text = ''.join(c for c in text[:30] if c.isalnum() or c in (
            ' ', '_')).strip().replace(' ', '_')
        return os.path.join(script_dir, f"tts_{speaker_name}_{timestamp}_{safe_text}.mp3")

    def speak(self, text: str, speaker_name: str = "Agent"):
        """Generate and play audio using gTTS, then save to file."""
        with self.lock:  # Ensure only one thread generates/plays audio at a time
            file_path = self._get_audio_filename(speaker_name, text)
            try:
                # Generate audio with gTTS
                tts = gTTS(text=text, lang='en')
                tts.save(file_path)

                # Play audio with pygame
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():  # Wait for playback to finish
                    pygame.time.Clock().tick(10)

                logger.success(f"TTS audio saved: {file_path}")
            except Exception as e:
                logger.error(f"Error in TTS generation/playback: {e}")
                raise

    async def speak_async(self, text: str, speaker_name: str = "Agent"):
        """Async wrapper for speak method."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text, speaker_name)
        await asyncio.sleep(0.1)  # Small delay to ensure smooth transitions


class Agent:
    def __init__(self, name: str, system_prompt: str, model: str = "llama3.1", session_id: str = "") -> None:
        self.name: str = name
        self.ollama: Ollama = Ollama(
            model=model, system=system_prompt, session_id=session_id, temperature=0.3)
        self.chat_history = self.ollama.chat_history
        # Initialize TTS engine (no voice_id needed for gTTS)
        self.tts = TTSEngine(rate=200 if name == "Interviewer" else 180)

    async def generate_response(self, external_message: str) -> str:
        content = ""
        async for chunk in self.ollama.stream_chat(query=external_message):
            content += chunk
        await self.tts.speak_async(f"{self.name}: {content}", speaker_name=self.name)
        return content

    def clear_history(self) -> None:
        """Reset the agent's conversation history."""
        self.chat_history.clear()


class Interviewer(Agent):
    def __init__(self, model: str = "llama3.1", **kwargs) -> None:
        super().__init__(
            name="Interviewer",
            system_prompt=(
                "You are a professional job interviewer for a software engineering position. "
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
                "Incorporate the candidate's previous response to maintain a natural conversation flow (e.g., 'Thank you for sharing that. [Next question]').\n"
                "Be polite, professional, and concise. After asking the final agenda topic (Closing), wait for the candidate's response.\n"
                "If the candidate raises questions or concerns, address them appropriately and then ask again if they have any further questions or concerns, repeating this process until they have no more.\n"
                "If the candidate indicates they have no further questions or concerns (e.g., 'No questions' or 'I'm good'), end the interview politely and include '[TERMINATE]' in your final message."
            ),
            model=model,
            **kwargs
        )


class Applicant(Agent):
    def __init__(self, model: str = "llama3.1", **kwargs) -> None:
        super().__init__(
            name="Applicant",
            system_prompt=(
                "You are a job applicant applying for a software engineering position. "
                "You have a strong background in Python, Java, and web development, with 3 years of experience. "
                "Respond to the interviewer's questions professionally, concisely, and with relevant details. "
                "If asked about weaknesses, be honest but frame them positively."
            ),
            model=model,
            **kwargs
        )


class Conversation:
    def __init__(self, agent1: Interviewer, agent2: Applicant, max_turns: int = 16) -> None:
        self.agent1: Interviewer = agent1
        self.agent2: Applicant = agent2
        self.max_turns: int = max_turns
        self.current_turn: int = agent1.chat_history.get_turn_count()

        messages = agent1.chat_history.get_messages()
        if messages and messages[-1]["role"] == "assistant":
            self.current_agent: Agent = self.agent2
        else:
            self.current_agent: Agent = self.agent1

    def switch_agent(self) -> None:
        """Switch the current agent between agent1 and agent2."""
        self.current_agent = self.agent2 if self.current_agent == self.agent1 else self.agent1

    async def run(self) -> None:
        """Run the conversation for the specified number of turns."""
        logger.orange("Starting Job Interview Conversation...\n")

        messages = self.agent1.chat_history.get_messages()
        if self.current_turn == 0:
            logger.info("Turn 1: Interviewer generating initial prompt")
            response = await self.agent1.generate_response("__START__")
        else:
            response = messages[-1]["content"]
            logger.info(
                f"Turn {self.current_turn + 1}: {self.current_agent.name}")
            logger.debug(
                f"Resuming conversation...\nLast message: {response}\n")

        self.switch_agent()
        self.current_turn += 1

        if "[TERMINATE]" in response:
            logger.orange("Interview terminated early by Interviewer")
            return

        while self.current_turn < self.max_turns:
            logger.info(
                f"Turn {self.current_turn + 1}: {self.current_agent.name}")
            response = await self.current_agent.generate_response(response)

            if self.current_turn + 1 >= self.max_turns and self.current_agent == self.agent1:
                response += " [TERMINATE]"

            logger.debug(f"{self.current_agent.name}: {response}\n")

            if self.current_agent == self.agent1 and "[TERMINATE]" in response:
                logger.info("Interview terminated by Interviewer")
                break

            self.switch_agent()
            self.current_turn += 1

    def reset(self) -> None:
        """Reset the conversation by clearing both agents' histories and resetting the agenda."""
        self.agent1.clear_history()
        self.agent2.clear_history()
        self.current_agent = self.agent1
        self.current_turn = self.agent1.chat_history.get_turn_count()


async def main() -> None:
    # Generate unique session IDs for each agent
    interviewer_session_id: str = generate_unique_hash()
    applicant_session_id: str = generate_unique_hash()
    max_turns = 20

    logger.success(f"Interviewer session id:\n{interviewer_session_id}\n")
    logger.success(f"Applicant session id:\n{applicant_session_id}\n")

    # Initialize agents with different session IDs (no voice IDs needed for gTTS)
    interviewer: Interviewer = Interviewer(session_id=interviewer_session_id)
    applicant: Applicant = Applicant(session_id=applicant_session_id)

    # Create and run conversation
    conversation: Conversation = Conversation(
        interviewer, applicant, max_turns=max_turns)
    await conversation.run()


if __name__ == "__main__":
    asyncio.run(main())
