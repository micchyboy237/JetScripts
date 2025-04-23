import asyncio
import os
from typing import Optional
import pyttsx3
import cv2
import numpy as np
from jet.data.utils import generate_unique_hash
from jet.llm.ollama.base import Ollama
from jet.logger.logger import CustomLogger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)


class TalkingHead:
    def __init__(self, face_image_path: str, voice_id: str = None):
        """Initialize the talking head with a face image and TTS engine."""
        self.face_image = cv2.imread(face_image_path)
        if self.face_image is None:
            raise FileNotFoundError(
                f"Face image not found at {face_image_path}")
        self.face_image = cv2.resize(self.face_image, (256, 256))
        self.engine = pyttsx3.init()
        if voice_id:
            self.engine.setProperty('voice', voice_id)
        self.engine.setProperty('rate', 150)  # Speech speed
        self.lip_sync_points = None  # Placeholder for lip landmarks
        self._initialize_lip_sync()

    def _initialize_lip_sync(self):
        """Initialize lip-sync points for animation (simplified)."""
        # For simplicity, define static lip region (x, y, width, height)
        # In a real application, use facial landmark detection (e.g., dlib or MediaPipe)
        lip_x, lip_y = 90, 160
        lip_w, lip_h = 80, 40
        self.lip_sync_points = (lip_x, lip_y, lip_w, lip_h)

    def animate_lip_sync(self, text: str, window_name: str):
        """Animate lip-sync while speaking the text."""
        # Start TTS in a separate thread to avoid blocking
        def speak():
            self.engine.say(text)
            self.engine.runAndWait()

        import threading
        tts_thread = threading.Thread(target=speak)
        tts_thread.start()

        # Simple lip animation: oscillate lip height based on a timer
        lip_x, lip_y, lip_w, lip_h = self.lip_sync_points
        frame = self.face_image.copy()
        start_time = cv2.getTickCount()

        while tts_thread.is_alive():
            # Simulate lip movement by scaling the lip region
            elapsed = (cv2.getTickCount() - start_time) / \
                cv2.getTickFrequency()
            # Oscillate between 0.7 and 1.3
            lip_scale = 1.0 + 0.3 * np.sin(10 * elapsed)
            new_lip_h = int(lip_h * lip_scale)

            # Draw the lip region (simplified as a rectangle)
            frame = self.face_image.copy()
            cv2.rectangle(
                frame,
                (lip_x, lip_y),
                (lip_x + lip_w, lip_y + new_lip_h),
                (255, 0, 0),  # Blue for visibility
                -1
            )

            # Display the animated frame
            cv2.imshow(window_name, frame)
            if cv2.waitKey(30) & 0xFF == 27:  # Exit on ESC
                break

        cv2.destroyWindow(window_name)
        tts_thread.join()

    def cleanup(self):
        """Clean up resources."""
        self.engine.stop()
        cv2.destroyAllWindows()


class Agent:
    def __init__(self, name: str, system_prompt: str, model: str = "llama3.1", session_id: str = "",
                 face_image_path: str = None, voice_id: str = None) -> None:
        self.name: str = name
        self.ollama: Ollama = Ollama(
            model=model, system=system_prompt, session_id=session_id, temperature=0.3)
        self.chat_history = self.ollama.chat_history
        self.talking_head = TalkingHead(
            face_image_path, voice_id) if face_image_path else None

    async def generate_response(self, external_message: str) -> str:
        """Generate a response and animate/speak it."""
        content = ""
        async for chunk in self.ollama.stream_chat(query=external_message):
            content += chunk

        if self.talking_head:
            self.talking_head.animate_lip_sync(content, f"{self.name} Talking")
        return content

    def clear_history(self) -> None:
        """Reset the agent's conversation history."""
        self.chat_history.clear()

    def cleanup(self):
        """Clean up talking head resources."""
        if self.talking_head:
            self.talking_head.cleanup()


class Interviewer(Agent):
    def __init__(self, model: str = "llama3.1", face_image_path: str = None, voice_id: str = None, **kwargs) -> None:
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
            face_image_path=face_image_path,
            voice_id=voice_id,
            **kwargs
        )


class Applicant(Agent):
    def __init__(self, model: str = "llama3.1", face_image_path: str = None, voice_id: str = None, **kwargs) -> None:
        super().__init__(
            name="Applicant",
            system_prompt=(
                "You are a job applicant applying for a software engineering position. "
                "You have a strong background in Python, Java, and web development, with 3 years of experience. "
                "Respond to the interviewer's questions professionally, concisely, and with relevant details. "
                "If asked about weaknesses, be honest but frame them positively."
            ),
            model=model,
            face_image_path=face_image_path,
            voice_id=voice_id,
            **kwargs
        )


class Conversation:
    def __init__(self, agent1: Interviewer, agent2: Applicant, max_turns: int = 16) -> None:
        self.agent1: Interviewer = agent1  # Interviewer
        self.agent2: Applicant = agent2  # Applicant
        self.max_turns: int = max_turns
        self.current_turn: int = agent1.chat_history.get_turn_count()

        # Determine the current agent based on the last message in the chat history
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
            self.cleanup()
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

        self.cleanup()

    def cleanup(self):
        """Clean up resources for both agents."""
        self.agent1.cleanup()
        self.agent2.cleanup()

    def reset(self) -> None:
        """Reset the conversation by clearing both agents' histories and resetting the agenda."""
        self.agent1.clear_history()
        self.agent2.clear_history()
        self.current_agent = self.agent1
        self.current_turn = self.agent1.chat_history.get_turn_count()


async def main() -> None:
    interviewer_session_id: str = generate_unique_hash()
    applicant_session_id: str = generate_unique_hash()
    max_turns = 20

    logger.success(f"Interviewer session id:\n{interviewer_session_id}\n")
    logger.success(f"Applicant session id:\n{applicant_session_id}\n")

    # Initialize agents with face images and distinct voices
    # Replace with actual paths to face images
    interviewer_face = os.path.join(script_dir, "interviewer_face.jpg")
    applicant_face = os.path.join(script_dir, "applicant_face.jpg")

    # Get available voices
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    interviewer_voice = voices[0].id if len(
        voices) > 0 else None  # Default voice
    applicant_voice = voices[1].id if len(
        voices) > 1 else None    # Different voice

    interviewer: Interviewer = Interviewer(
        session_id=interviewer_session_id,
        face_image_path=interviewer_face,
        voice_id=interviewer_voice
    )
    applicant: Applicant = Applicant(
        session_id=applicant_session_id,
        face_image_path=applicant_face,
        voice_id=applicant_voice
    )

    conversation: Conversation = Conversation(
        interviewer, applicant, max_turns=max_turns)
    await conversation.run()


if __name__ == "__main__":
    asyncio.run(main())
