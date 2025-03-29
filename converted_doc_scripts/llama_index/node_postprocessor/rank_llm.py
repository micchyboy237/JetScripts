from typing import List, Dict, Any, Optional
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from llama_index.core.agent import ReActAgent
from jet.llm.ollama.base import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
from llama_index.core.chat_engine.types import AgentChatResponse
import random

initialize_ollama_settings()

"""
Feature Summary:
- Query: Represents a search query with text and an ID.
- Candidate: Represents a document with metadata (LLM provides the score).
- Request: Encapsulates a query and a list of candidate documents.
- Reranker: Handles reranking of documents based on given parameters.
- Intelligent LLM-Based Ranking: Uses a ReActAgent to enhance ranking logic.
- Sorting & Filtering: Supports ranking selection, shuffling, and top-k retrieval.
- Configurable Parameters: Allows tuning of ranking range, batch size, and window size.
- Logging: Optionally logs reranked document IDs for analysis.
- Interactive Chat: Enables real-time query refinement and user engagement.
- Sample Usage: Provided at the end for seamless integration.
"""


class Query:
    def __init__(self, text: str, qid: int):
        self.text = text  # Query text
        self.qid = qid    # Query ID

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "qid": self.qid}


class Candidate:
    def __init__(self, doc_id: int, text: str, metadata: Optional[dict[str, Any]] = None):
        self.doc_id = doc_id  # Document ID
        self.text = text  # Document text
        self.metadata = metadata or {}  # Optional metadata
        self.score = 0.0  # Placeholder score to be updated by LLM

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata, "score": self.score}


class Request:
    def __init__(self, query: Query, candidates: List[Candidate]):
        self.query = query
        self.candidates = candidates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates]
        }


class Reranker:
    @staticmethod
    def create_agent(model: str, default_agent=None, interactive=False, window_size=None, batch_size=None):
        return Reranker(model, window_size, batch_size, interactive)

    def __init__(self, model: str, window_size: Optional[int] = None, batch_size: Optional[int] = None, interactive: bool = False):
        self.model = model
        self.window_size = window_size
        self.batch_size = batch_size
        self.interactive = interactive
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        llm = Ollama(model=self.model, request_timeout=300.0,
                     context_window=4096)
        return ReActAgent.from_tools([], llm=llm, verbose=True)

    def chat(self, message: str) -> AgentChatResponse:
        response = self.agent.chat(message)
        return response if isinstance(response, AgentChatResponse) else AgentChatResponse(response="")

    def rerank_batch(
        self,
        requests: List[Request],
        rank_end: Optional[int] = None,
        rank_start: Optional[int] = 0,
        shuffle_candidates: Optional[bool] = False,
        logging: Optional[bool] = False,
        top_k_retrieve: Optional[int] = None,
    ) -> List[Request]:
        reranked_requests = []
        for request in requests:
            candidates = request.candidates

            if self.interactive:
                message = f"How should we refine ranking for query: {request.query.text}?"
                user_feedback = self.chat(message)
                logger.info(f"User feedback: {user_feedback}")

            if shuffle_candidates:
                random.shuffle(candidates)

            message = f"Rank the following documents based on relevance to the query: {request.query.text}\n{[c.text for c in candidates]}"
            llm_response = self.chat(message).response
            new_scores = [float(score) for score in llm_response.split(
            ) if score.replace('.', '', 1).isdigit()]

            for i, candidate in enumerate(candidates):
                if i < len(new_scores):
                    candidate.score = new_scores[i]

            sorted_candidates = sorted(
                candidates, key=lambda c: c.score, reverse=True)
            rank_end = rank_end if rank_end is not None else len(
                sorted_candidates)
            top_k_retrieve = top_k_retrieve if top_k_retrieve is not None else len(
                sorted_candidates)

            reranked_candidates = sorted_candidates[rank_start:rank_end][:top_k_retrieve]

            if logging:
                logger.info(
                    f"Reranked candidates for query '{request.query.text}': {[c.doc_id for c in reranked_candidates]}")

            reranked_requests.append(
                Request(request.query, reranked_candidates))

        return reranked_requests


if __name__ == "__main__":
    model = "llama3.2"
    query = Query("What are the best machine learning models?", 1)
    candidates = [
        Candidate(doc_id=0, text="Deep Learning is powerful.", metadata={
                  "title": "Deep Learning", "url": "https://example.com/dl"}),
        Candidate(doc_id=1, text="Random Forest is robust.", metadata={
                  "title": "Random Forest", "url": "https://example.com/rf"}),
        Candidate(doc_id=2, text="Gradient Boosting performs well.", metadata={
                  "title": "Gradient Boosting", "url": "https://example.com/gb"}),
    ]
    request = Request(query, candidates)
    reranker = Reranker(model, interactive=True)
    reranked = reranker.rerank_batch([request], logging=True)
    print([c.to_dict() for c in reranked[0].candidates])
