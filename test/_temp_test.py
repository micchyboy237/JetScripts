import asyncio
import os
from typing import Optional
from jet.data.utils import generate_unique_hash
from jet.llm.ollama.base import Ollama
from jet.logger.logger import CustomLogger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)


class Agent:
    def __init__(self, name: str, system_prompt: str, model: str = "llama3.1", session_id: str = "") -> None:
        self.name: str = name
        self.ollama: Ollama = Ollama(
            model=model, system=system_prompt, session_id=session_id, temperature=0.75)
        self.chat_history = self.ollama.chat_history

    async def generate_response(self, external_message: str) -> str:
        """Generate a response using Ollama's stream_chat."""
        content = ""
        async for chunk in self.ollama.stream_chat(query=external_message):
            content += chunk
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
                "Use the chat history to determine which agenda topic to address next, ensuring you progress through the topics in order. "
                "Incorporate the candidate's previous response to maintain a natural conversation flow (e.g., 'Thank you for sharing that. [Next question]'). "
                "Be polite, professional, and concise. After the final agenda topic, include '[TERMINATE]' in your response."
            ),
            model=model,
            **kwargs
        )

    async def generate_response(self, query: str) -> str:
        """Generate a response based on the agenda in the system prompt and chat history."""
        content = ""
        async for chunk in self.ollama.stream_chat(query=query):
            content += chunk
        return content


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
        self.agent1: Interviewer = agent1  # Interviewer
        self.agent2: Applicant = agent2  # Applicant
        self.max_turns: int = max_turns
        self.current_turn: int = agent1.chat_history.get_turn_count()

        # Determine the current agent based on the last message in the chat history
        messages = agent1.chat_history.get_messages()
        if messages and messages[-1]["role"] == "assistant":
            # Last message was from Interviewer, so Applicant should respond next
            self.current_agent: Agent = self.agent2
        else:
            # Last message was from Applicant (or no messages), so Interviewer should respond
            self.current_agent: Agent = self.agent1

    def switch_agent(self) -> None:
        """Switch the current agent between agent1 and agent2."""
        self.current_agent = self.agent2 if self.current_agent == self.agent1 else self.agent1

    async def run(self) -> None:
        """Run the conversation for the specified number of turns."""
        print("Starting Job Interview Conversation...\n")

        messages = self.agent1.chat_history.get_messages()
        if self.current_turn == 0:
            # No messages, dynamically generate Interviewer's first question
            logger.info("Turn 1: Interviewer generating initial prompt")
            response = await self.agent1.generate_response("__START__")
            logger.debug(f"{self.agent1.name} response: {response}")
            print(f"{self.agent1.name}: {response}\n")
        else:
            # Resume from the last message
            response = messages[-1]["content"]
            logger.info(
                f"Turn {self.current_turn + 1}: {self.current_agent.name}")
            logger.debug(f"Resuming with last message: {response}")
            print(f"Resuming conversation...\nLast message: {response}\n")

        # Switch here after first response
        self.switch_agent()
        self.current_turn += 1

        if "[TERMINATE]" in response:
            logger.info("Interview terminated early by Interviewer")
            return

        while self.current_turn < self.max_turns:
            logger.info(
                f"Turn {self.current_turn + 1}: {self.current_agent.name}")
            response = await self.current_agent.generate_response(response)

            if self.current_turn + 1 >= self.max_turns and self.current_agent == self.agent1:
                response += " [TERMINATE]"

            logger.debug(f"{self.current_agent.name} response: {response}")
            print(f"{self.current_agent.name}: {response}\n")

            if self.current_agent == self.agent1 and "[TERMINATE]" in response:
                logger.info("Interview terminated by Interviewer")
                break

            self.switch_agent()
            self.current_turn += 1

        def reset(self) -> None:
            """Reset the conversation by clearing both agents' histories and resetting the agenda."""
            self.agent1.clear_history()
            self.agent2.clear_history()
            self.current_agent = self.agent1  # Reset to Interviewer
            self.current_turn = self.agent1.chat_history.get_turn_count()


async def main() -> None:
    session_id: str = generate_unique_hash()
    # session_id: str = "89689a82-5d40-474e-a817-d14ec4a45124"
    max_turns = 16

    # Initialize agents
    interviewer: Interviewer = Interviewer(session_id=session_id)
    applicant: Applicant = Applicant(session_id=session_id)

    # Create and run conversation
    conversation: Conversation = Conversation(
        interviewer, applicant, max_turns=max_turns)
    await conversation.run()

if __name__ == "__main__":
    asyncio.run(main())
