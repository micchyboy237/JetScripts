import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from helpers import (
    setup_config, load_json_data, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response, save_file,
    DATA_DIR, DOCS_PATH, cosine_similarity
)
from jet.logger import CustomLogger


class ChunkSizeSelector:
    def __init__(self, script_path: str):
        self.script_dir, self.generated_dir, self.log_file, self.logger = setup_config(
            script_path)
        self.mlx, self.embed_func = initialize_mlx(self.logger)
        self.chunk_sizes = [128, 256, 512]

    def extract_text(self, formatted_chunks: List[str]) -> str:
        """Extract text content from formatted chunks."""
        return " ".join([chunk.split("\n\n")[-1] for chunk in formatted_chunks])

    def chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        """Split text into chunks of specified size with overlap."""
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunks.append(text[i:i + size])
        return chunks

    def generate_chunk_dict(self, text: str) -> Dict[int, List[str]]:
        """Generate dictionary of chunks for each chunk size."""
        return {
            size: self.chunk_text(text, size, size // 5)
            for size in self.chunk_sizes
        }

    def generate_chunk_embeddings(self, chunk_dict: Dict[int, List[str]]) -> Dict[int, List[np.ndarray]]:
        """Generate embeddings for all chunks."""
        return {
            size: generate_embeddings(chunks, self.embed_func, self.logger)
            for size, chunks in tqdm(chunk_dict.items(), desc="Generating Embeddings")
        }

    def retrieve_relevant_chunks(
        self,
        query: str,
        text_chunks: List[str],
        chunk_embeddings: List[np.ndarray],
        k: int = 5
    ) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query."""
        query_embedding = self.embed_func(query)
        similarities = [cosine_similarity(
            query_embedding, emb) for emb in chunk_embeddings]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [
            {
                "id": f"chunk_{i}",
                "rank": idx + 1,
                "doc_index": i,
                "score": similarities[i],
                "text": text_chunks[i]
            }
            for idx, i in enumerate(top_indices)
        ]

    def evaluate_response(
        self,
        question: str,
        response: str,
        true_answer: str
    ) -> Tuple[float, float]:
        """Evaluate response faithfulness and relevancy."""
        FAITHFULNESS_PROMPT_TEMPLATE = (
            "Evaluate how faithful the AI response is to the true answer. "
            "Score 1.0 for complete faithfulness, 0.5 for partial, 0.0 for none.\n"
            "Question: {question}\nAIShi Response: {response}\nTrue Answer: {true_answer}\n"
            "Return ONLY the numerical score (1.0, 0.5, or 0.0)."
        )
        RELEVANCY_PROMPT_TEMPLATE = (
            "Evaluate how relevant the AI response is to the question. "
            "Score 1.0 for complete relevancy, 0.5 for partial, 0.0 for none.\n"
            "Question: {question}\nAI Response: {response}\nTrue Answer: {true_answer}\n"
            "Return ONLY the numerical score (1.0, 0.5, or 0.0)."
        )
        faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            question=question,
            response=response,
            true_answer=true_answer
        )
        relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
            question=question,
            response=response,
            true_answer=true_answer
        )
        faithfulness_response = self.mlx.chat(
            [
                {"role": "system",
                    "content": "You are an objective evaluator. Return ONLY the numerical score (1.0, 0.5, or 0.0)."},
                {"role": "user", "content": faithfulness_prompt}
            ]
        )
        relevancy_response = self.mlx.chat(
            [
                {"role": "system",
                    "content": "You are an objective evaluator. Return ONLY the numerical score (1.0, 0.5, or 0.0)."},
                {"role": "user", "content": relevancy_prompt}
            ]
        )

        def parse_score(response_content: str) -> float:
            content = response_content.strip()
            self.logger.debug(f"Raw score response: {content}")
            try:
                score = float(content)
                if score not in [0.0, 0.5, 1.0]:
                    self.logger.debug(
                        f"Invalid score {score}, defaulting to 0.0")
                    return 0.0
                return score
            except ValueError:
                self.logger.debug("Could not parse score, defaulting to 0.0")
                return 0.0

        faithfulness_score = parse_score(faithfulness_response["content"])
        relevancy_score = parse_score(relevancy_response["content"])
        return faithfulness_score, relevancy_score

    def run(self):
        """Main execution flow."""
        self.logger.info("Loading JSON data")
        formatted_chunks, _ = load_json_data(DOCS_PATH, self.logger)

        extracted_text = self.extract_text(formatted_chunks)
        self.logger.debug(extracted_text[:500])

        self.logger.info("Chunking text with different sizes")
        text_chunks_dict = self.generate_chunk_dict(extracted_text)

        for size, chunks in text_chunks_dict.items():
            self.logger.debug(
                f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")

        save_file({"chunk_sizes": text_chunks_dict},
                  f"{self.generated_dir}/text_chunks.json")
        self.logger.info(
            f"Saved text chunks to {self.generated_dir}/text_chunks.json")

        self.logger.info("Generating embeddings for chunks")
        chunk_embeddings_dict = self.generate_chunk_embeddings(
            text_chunks_dict)

        self.logger.info("Processing validation data")
        validation_data = load_validation_data(
            f"{DATA_DIR}/val.json", self.logger)
        query = validation_data[3]['question']

        retrieved_chunks_dict = {
            size: self.retrieve_relevant_chunks(
                query, text_chunks_dict[size], chunk_embeddings_dict[size])
            for size in self.chunk_sizes
        }

        self.logger.debug(retrieved_chunks_dict[256])
        save_file(retrieved_chunks_dict,
                  f"{self.generated_dir}/retrieved_chunks.json")
        self.logger.info(
            f"Saved retrieved chunks to {self.generated_dir}/retrieved_chunks.json")

        self.logger.info("Generating AI responses")
        system_prompt = (
            "You are an AI assistant that strictly answers based on the given context. "
            "If the answer cannot be derived directly from the provided context, "
            "respond with: 'I do not have enough information to answer that.'"
        )

        ai_responses_dict = {
            size: generate_ai_response(
                query, system_prompt, retrieved_chunks_dict[size], self.mlx, self.logger
            )
            for size in self.chunk_sizes
        }

        self.logger.debug(ai_responses_dict[256])
        save_file(ai_responses_dict, f"{self.generated_dir}/ai_responses.json")
        self.logger.info(
            f"Saved AI responses to {self.generated_dir}/ai_responses.json")

        self.logger.info("Evaluating responses")
        true_answer = validation_data[3]['ideal_answer']
        evaluation_scores = {}

        for size in [256, 128]:
            faithfulness, relevancy = self.evaluate_response(
                query, ai_responses_dict[size], true_answer
            )
            evaluation_scores[f"chunk_size_{size}"] = {
                "faithfulness": faithfulness,
                "relevancy": relevancy
            }
            self.logger.debug(
                f"\nFaithfulness Score (Chunk Size {size}): {faithfulness}")
            self.logger.debug(
                f"Relevancy Score (Chunk Size {size}): {relevancy}")

        save_file(evaluation_scores,
                  f"{self.generated_dir}/evaluation_scores.json")
        self.logger.info(
            f"Saved evaluation scores to {self.generated_dir}/evaluation_scores.json")
        self.logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    selector = ChunkSizeSelector(__file__)
    selector.run()
