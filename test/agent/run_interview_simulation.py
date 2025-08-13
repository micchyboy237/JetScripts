from typing import Generator, List, Tuple
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import GenerationResponse
from mlx_lm.sample_utils import make_sampler
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    """Manages a knowledge base with FAISS for retrieval."""

    def __init__(self, documents: List[str]):
        """Initialize with documents and create FAISS index."""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents
        self.index = faiss.IndexFlatL2(384)  # MiniLM embedding size
        self.embeddings = self.model.encode(documents)
        self.index.add(np.array(self.embeddings, dtype=np.float32))

    def retrieve(self, query: str, k: int = 2) -> List[str]:
        """Retrieve top-k relevant documents for a query."""
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(
            np.array(query_embedding, dtype=np.float32), k)
        return [self.documents[i] for i in indices[0]]


class InterviewerAgent:
    """Agent that simulates an interviewer with RAG and resume knowledge."""

    def __init__(self, knowledge_base: KnowledgeBase, model_name: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"):
        """Initialize with MLX model and knowledge base."""
        self.model, self.tokenizer = load(model_name)
        self.knowledge_base = knowledge_base
        self.system_prompt = (
            "You are a professional interviewer for a software engineer role. "
            "Follow this agenda: 1) Greet the candidate and introduce the role, "
            "2) Ask about technical skills (e.g., Python, SQL) based on the candidate's resume, "
            "3) Ask about teamwork and problem-solving, "
            "4) Ask if the candidate has questions or concerns. "
            "Use the provided resume and job context to ask specific, relevant questions. "
            "End with '[TERMINATE]' if the candidate has no further questions."
        )

    def generate_response(self, chat_history: List[Tuple[str, str]]) -> Generator[GenerationResponse, None, None]:
        """Generate interviewer's response using RAG with resume context."""
        history_str = "\n".join(
            [f"{role}: {msg}" for role, msg in chat_history])
        query = history_str or "software engineer resume skills"
        context = self.knowledge_base.retrieve(query)
        prompt = (
            f"{self.system_prompt}\n\nContext (resume and job details): {'; '.join(context)}\n\n"
            f"Chat History:\n{history_str}\n\nInterviewer:"
        )
        sampler = make_sampler(temp=0.7)
        yield from stream_generate(self.model, self.tokenizer, prompt=prompt,
                                   max_tokens=100, sampler=sampler)


class ApplicantAgent:
    """Agent that simulates an applicant with RAG."""

    def __init__(self, knowledge_base: KnowledgeBase, model_name: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"):
        """Initialize with MLX model and knowledge base."""
        self.model, self.tokenizer = load(model_name)
        self.knowledge_base = knowledge_base
        self.system_prompt = (
            "You are a job applicant with 3 years of experience in Python and SQL. "
            "Respond concisely and professionally, using the provided resume context to tailor your answers. "
            "In the final stage, if asked about questions, respond with 'I have no further questions.'"
        )

    def generate_response(self, chat_history: List[Tuple[str, str]]) -> Generator[GenerationResponse, None, None]:
        """Generate applicant's response using RAG."""
        history_str = "\n".join(
            [f"{role}: {msg}" for role, msg in chat_history])
        query = history_str.split(
            "\n")[-1] if history_str else "software engineer skills"
        context = self.knowledge_base.retrieve(query)
        prompt = (
            f"{self.system_prompt}\n\nContext (resume and job details): {'; '.join(context)}\n\n"
            f"Chat History:\n{history_str}\n\nApplicant:"
        )
        sampler = make_sampler(temp=0.7)
        yield from stream_generate(self.model, self.tokenizer, prompt=prompt,
                                   max_tokens=100, sampler=sampler)


class Conversation:
    """Manages the back-and-forth interaction between interviewer and applicant."""

    def __init__(self, knowledge_base: KnowledgeBase, max_turns: int = 8):
        """Initialize conversation with knowledge base and max turns."""
        self.interviewer = InterviewerAgent(knowledge_base)
        self.applicant = ApplicantAgent(knowledge_base)
        self.chat_history: List[Tuple[str, str]] = []
        self.max_turns = max_turns

    def run(self) -> Generator[Tuple[str, str], None, None]:
        """Run the interview simulation."""
        for turn in range(self.max_turns):
            if turn % 2 == 0:
                text = ""
                for response in self.interviewer.generate_response(self.chat_history):
                    logger.teal(response.text, flush=True)
                    text += response.text
                self.chat_history.append(("Interviewer", text))
                yield "Interviewer", text
                if "[TERMINATE]" in text:
                    break
            else:
                text = ""
                for response in self.applicant.generate_response(self.chat_history):
                    logger.teal(response.text, flush=True)
                    text += response.text
                self.chat_history.append(("Applicant", text))
                yield "Applicant", text


# Example usage
if __name__ == "__main__":
    import os
    import shutil
    from jet.file.utils import load_file, save_file

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    documents = [
        "Software Engineer role requiring 3+ years of Python and SQL experience.",
        "Candidate resume: 3 years of Python experience with pandas for data pipelines, 2 years of SQL for database management.",
        "Common question: Describe a challenging Python project you worked on.",
        "Teamwork is critical for collaborative software development."
    ]
    knowledge_base = KnowledgeBase(documents)
    conversation = Conversation(knowledge_base)

    history = conversation.run()
    conversation_history = []
    print("Interview Simulation with RAG and Resume Knowledge:")
    for role, msg in history:
        conversation_history.append({
            "role": role,
            "content": msg
        })
        save_file(conversation_history,
                  f"{output_dir}/conversation_history.json")
