import asyncio
import os
from typing import Optional
from jet.llm.ollama.base import Ollama
from jet.logger.logger import CustomLogger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)


class Agent:
    def __init__(self, name: str, system_prompt: str, model: str = "llama3.1") -> None:
        self.name: str = name
        self.ollama: Ollama = Ollama(
            model=model, system=system_prompt, session_id=f"{name}_{model}")

    async def generate_response(self, external_message: Optional[str] = None) -> str:
        """Generate a response using Ollama's stream_chat."""
        content = ""
        async for chunk in self.ollama.stream_chat(query=external_message or "Continue the conversation"):
            content += chunk
        return content

    def clear_history(self) -> None:
        """Reset the agent's conversation history."""
        self.ollama.chat_history.clear()


class Interviewer(Agent):
    def __init__(self, model: str = "llama3.1") -> None:
        super().__init__(
            name="Interviewer",
            system_prompt=(
                "You are a professional job interviewer for a software engineering position. "
                "Ask clear, relevant questions to assess the candidate's technical skills, experience, "
                "and problem-solving abilities. Be polite, structured, and focus on one question at a time. "
                "Wait for the candidate's response before asking the next question."
            ),
            model=model
        )


class Applicant(Agent):
    def __init__(self, model: str = "llama3.1") -> None:
        super().__init__(
            name="Applicant",
            system_prompt=(
                "You are a job applicant applying for a software engineering position. "
                "You have a strong background in Python, Java, and web development, with 3 years of experience. "
                "Respond to the interviewer's questions professionally, concisely, and with relevant details. "
                "If asked about weaknesses, be honest but frame them positively."
            ),
            model=model
        )


class Conversation:
    def __init__(self, agent1: Agent, agent2: Agent, max_turns: int = 6) -> None:
        self.agent1: Agent = agent1  # Interviewer
        self.agent2: Agent = agent2  # Applicant
        self.max_turns: int = max_turns
        self.current_agent: Agent = agent1

    def switch_agent(self) -> None:
        """Switch the current agent between agent1 and agent2."""
        self.current_agent = self.agent2 if self.current_agent == self.agent1 else self.agent1

    async def run(self) -> None:
        """Run the conversation for the specified number of turns."""
        print("Starting Job Interview Conversation...\n")

        for turn in range(self.max_turns):
            # Log turn and agent
            logger.info(f"Turn {turn + 1}: {self.current_agent.name}")

            # Agent1 (Interviewer) starts, then alternates
            response: str = await self.current_agent.generate_response()
            logger.debug(f"{self.current_agent.name} response: {response}")
            print(f"{self.current_agent.name}: {response}\n")

            # Pass the response to the other agent
            self.switch_agent()

            # Log turn and agent
            logger.info(f"Turn {turn + 1}: {self.current_agent.name}")
            response = await self.current_agent.generate_response(response)
            logger.debug(f"{self.current_agent.name} response: {response}")
            print(f"{self.current_agent.name}: {response}\n")

            # Switch back for the next turn
            self.switch_agent()

    def reset(self) -> None:
        """Reset the conversation by clearing both agents' histories."""
        self.agent1.clear_history()
        self.agent2.clear_history()
        self.current_agent = self.agent1


async def main() -> None:
    # Initialize agents
    interviewer: Interviewer = Interviewer()
    applicant: Applicant = Applicant()

    # Create and run conversation
    conversation: Conversation = Conversation(
        interviewer, applicant, max_turns=6)
    await conversation.run()

if __name__ == "__main__":
    asyncio.run(main())
