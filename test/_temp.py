from typing import List, Tuple
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors


class InterviewerAgent:
    """Agent that simulates an interviewer following a structured agenda."""

    def __init__(self, model_name: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"):
        """Initialize with MLX model."""
        self.model, self.tokenizer = load(model_name)
        self.system_prompt = (
            "You are a professional interviewer conducting a software engineer interview. "
            "Follow this agenda: 1) Greet the candidate and introduce the role, "
            "2) Ask about technical skills (e.g., Python, SQL), "
            "3) Ask about teamwork and problem-solving, "
            "4) Ask if the candidate has questions or concerns. "
            "Generate one concise response per turn, relevant to the current agenda step based on chat history. "
            "If the candidate indicates no further questions in the final step, end with '[TERMINATE]'."
        )

    def generate_response(self, chat_history: List[Tuple[str, str]]) -> str:
        """Generate interviewer's response based on chat history."""
        history_str = "\n".join(
            [f"{role}: {msg}" for role, msg in chat_history])
        prompt = f"{self.system_prompt}\n\nChat History:\n{history_str}\n\nInterviewer:"
        sampler = make_sampler(temp=0.7)
        response = generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=100, sampler=sampler, verbose=True)
        return response.strip()


class ApplicantAgent:
    """Agent that simulates a job applicant responding to interview questions."""

    def __init__(self, model_name: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"):
        """Initialize with MLX model."""
        self.model, self.tokenizer = load(model_name)
        self.system_prompt = (
            "You are a job applicant for a software engineer position with 3 years of experience in Python and SQL. "
            "Respond concisely and professionally to the interviewer's questions. "
            "In the final stage, if asked about questions or concerns, respond with 'I have no further questions.'"
        )

    def generate_response(self, chat_history: List[Tuple[str, str]]) -> str:
        """Generate applicant's response based on chat history."""
        history_str = "\n".join(
            [f"{role}: {msg}" for role, msg in chat_history])
        prompt = f"{self.system_prompt}\n\nChat History:\n{history_str}\n\nApplicant:"
        sampler = make_sampler(temp=0.7)
        response = generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=100, sampler=sampler, verbose=True)
        return response.strip()


class Conversation:
    """Manages the back-and-forth interaction between interviewer and applicant."""

    def __init__(self, max_turns: int = 8):
        """Initialize conversation with max turns."""
        self.interviewer = InterviewerAgent()
        self.applicant = ApplicantAgent()
        self.chat_history: List[Tuple[str, str]] = []
        self.max_turns = max_turns

    def run(self) -> List[Tuple[str, str]]:
        """Run the interview simulation."""
        for turn in range(self.max_turns):
            # Interviewer goes first on odd turns (0-based index)
            if turn % 2 == 0:
                response = self.interviewer.generate_response(
                    self.chat_history)
                self.chat_history.append(("Interviewer", response))
                if "[TERMINATE]" in response:
                    break
            # Applicant responds on even turns
            else:
                response = self.applicant.generate_response(self.chat_history)
                self.chat_history.append(("Applicant", response))
        return self.chat_history


# Example usage
if __name__ == "__main__":
    conversation = Conversation()
    history = conversation.run()
    print("Interview Simulation:")
    for role, msg in history:
        print(f"{role}: {msg}")
