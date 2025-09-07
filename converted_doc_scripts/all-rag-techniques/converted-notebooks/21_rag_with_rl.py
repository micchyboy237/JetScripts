import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def split_into_chunks(text: str, chunk_size: int = 100) -> List[str]:
    """Split text into chunks based on word count."""
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing and keeping alphanumeric characters."""
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text


def preprocess_chunks(chunks: List[str]) -> List[str]:
    """Preprocess a list of chunks."""
    return [preprocess_text(chunk["text"]) for chunk in chunks]


def add_to_vector_store(vector_store: SimpleVectorStore, embeddings: List[np.ndarray], chunks: List[str]) -> None:
    """Add embeddings and chunks to vector store."""
    for embedding, chunk in zip(embeddings, chunks):
        vector_store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"chunk_index": len(vector_store.texts)}
        )


def construct_prompt(query: str, context_chunks: List[str]) -> str:
    """Construct prompt for response generation."""
    context = "\n".join(context_chunks)
    system_message = (
        "You are a helpful assistant. Only use the provided context to answer the question. "
        "If the context doesn't contain the information needed, say 'I don't have enough information to answer this question.'"
    )
    prompt = f"System: {system_message}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    return prompt


def basic_rag_pipeline(query: str, vector_store: SimpleVectorStore, embed_func, mlx, model: str = "llama-3.2-3b-instruct-4bit") -> str:
    """Run basic RAG pipeline."""
    query_embedding = embed_func(query)
    relevant_chunks = vector_store.search(query_embedding, top_k=5)
    context = [chunk["text"] for chunk in relevant_chunks]
    prompt = construct_prompt(query, context)
    response = generate_ai_response(
        query,
        prompt,
        relevant_chunks,
        mlx,
        logger,
        model=model,
        max_tokens=512,
        temperature=1.0
    )
    return response


def define_state(
    query: str,
    context_chunks: List[str],
    rewritten_query: str = None,
    previous_responses: List[str] = None,
    previous_rewards: List[float] = None
) -> dict:
    """Define RL state."""
    return {
        "original_query": query,
        "current_query": rewritten_query if rewritten_query else query,
        "context": context_chunks,
        "previous_responses": previous_responses if previous_responses else [],
        "previous_rewards": previous_rewards if previous_rewards else []
    }


def define_action_space() -> List[str]:
    """Define RL action space."""
    return ["rewrite_query", "expand_context", "filter_context", "generate_response"]


def calculate_reward(response: str, ground_truth: str, embed_func) -> float:
    """Calculate reward based on cosine similarity."""
    response_embedding = embed_func(response)
    ground_truth_embedding = embed_func(ground_truth)
    return np.dot(response_embedding, ground_truth_embedding) / (
        np.linalg.norm(response_embedding) *
        np.linalg.norm(ground_truth_embedding)
    )


def rewrite_query(
    query: str,
    context_chunks: List[str],
    mlx,
    model: str = "llama-3.2-3b-instruct-4bit",
    max_tokens: int = 100,
    temperature: float = 0.3
) -> str:
    """Rewrite query for better precision."""
    system_prompt = "You are a helpful assistant. Rewrite the query to make it more precise and contextually relevant, based on the provided context."
    context = "\n".join(context_chunks)
    user_prompt = f"Context:\n{context}\n\nOriginal Query: {query}\n\nRewritten Query:"
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def expand_context(query: str, current_chunks: List[str], vector_store: SimpleVectorStore, embed_func, top_k: int = 3) -> List[str]:
    """Expand context with additional relevant chunks."""
    query_embedding = embed_func(query)
    additional_chunks = vector_store.search(
        query_embedding, top_k=top_k + len(current_chunks))
    new_chunks = [chunk["text"]
                  for chunk in additional_chunks if chunk["text"] not in current_chunks]
    return current_chunks + new_chunks[:top_k]


def filter_context(query: str, context_chunks: List[str], embed_func) -> List[str]:
    """Filter context to keep most relevant chunks."""
    if not context_chunks:
        return []
    query_embedding = embed_func(query)
    chunk_embeddings = [embed_func(chunk) for chunk in context_chunks]
    relevance_scores = [
        np.dot(query_embedding, chunk_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
        ) for chunk_emb in chunk_embeddings
    ]
    sorted_chunks = [x for _, x in sorted(
        zip(relevance_scores, context_chunks), reverse=True)]
    return sorted_chunks[:min(5, len(sorted_chunks))]


def policy_network(
    state: dict,
    action_space: List[str],
    epsilon: float = 0.2
) -> str:
    """Select action using epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    if not state["previous_responses"]:
        return "rewrite_query"
    if state["previous_rewards"] and max(state["previous_rewards"]) < 0.7:
        return "expand_context"
    if len(state["context"]) > 5:
        return "filter_context"
    return "generate_response"


def rl_step(
    state: dict,
    action_space: List[str],
    ground_truth: str,
    vector_store: SimpleVectorStore,
    embed_func,
    mlx,
    model: str = "llama-3.2-3b-instruct-4bit"
) -> tuple[dict, str, float, str]:
    """Execute one RL step."""
    action = policy_network(state, action_space)
    response = None
    reward = 0
    if action == "rewrite_query":
        rewritten_query = rewrite_query(
            state["original_query"], state["context"], mlx, model)
        state["current_query"] = rewritten_query
        query_embedding = embed_func(rewritten_query)
        new_context = [chunk["text"]
                       for chunk in vector_store.search(query_embedding, top_k=5)]
        state["context"] = new_context
    elif action == "expand_context":
        state["context"] = expand_context(
            state["current_query"], state["context"], vector_store, embed_func)
    elif action == "filter_context":
        state["context"] = filter_context(
            state["current_query"], state["context"], embed_func)
    elif action == "generate_response":
        prompt = construct_prompt(state["current_query"], state["context"])
        response = generate_ai_response(
            state["current_query"],
            prompt,
            [{"text": chunk} for chunk in state["context"]],
            mlx,
            logger,
            model=model,
            max_tokens=512,
            temperature=1.0
        )
        reward = calculate_reward(response, ground_truth, embed_func)
        state["previous_responses"].append(response)
        state["previous_rewards"].append(reward)
    return state, action, reward, response


def initialize_training_params() -> Dict[str, Union[float, int]]:
    """Initialize RL training parameters."""
    return {
        "learning_rate": 0.01,
        "num_episodes": 100,
        "discount_factor": 0.99
    }


def update_policy(
    policy: Dict[str, Dict[str, Union[float, str]]],
    state: Dict[str, object],
    action: str,
    reward: float,
    learning_rate: float
) -> Dict[str, Dict[str, Union[float, str]]]:
    """Update RL policy."""
    policy[state["original_query"]] = {
        "action": action,
        "reward": reward
    }
    return policy


def training_loop(
    query_text: str,
    ground_truth: str,
    vector_store: SimpleVectorStore,
    embed_func,
    mlx,
    params: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[Dict[str, Dict[str, Union[float, str]]], List[float], List[List[str]], Optional[str]]:
    """Run RL training loop."""
    if params is None:
        params = initialize_training_params()
    rewards_history = []
    actions_history = []
    policy = {}
    action_space = define_action_space()
    best_response = None
    best_reward = -1
    simple_response = basic_rag_pipeline(
        query_text, vector_store, embed_func, mlx)
    simple_reward = calculate_reward(simple_response, ground_truth, embed_func)
    logger.debug(f"Simple RAG reward: {simple_reward:.4f}")
    for episode in range(params["num_episodes"]):
        query_embedding = embed_func(query_text)
        context_chunks = [chunk["text"]
                          for chunk in vector_store.search(query_embedding, top_k=5)]
        state = define_state(query_text, context_chunks)
        episode_reward = 0
        episode_actions = []
        for step in range(10):
            state, action, reward, response = rl_step(
                state, action_space, ground_truth, vector_store, embed_func, mlx
            )
            episode_actions.append(action)
            if response:
                episode_reward = reward
                if reward > best_reward:
                    best_reward = reward
                    best_response = response
                break
        rewards_history.append(episode_reward)
        actions_history.append(episode_actions)
        if episode % 5 == 0:
            logger.debug(
                f"Episode {episode}: Reward = {episode_reward:.4f}, Actions = {episode_actions}")
    improvement = best_reward - simple_reward
    logger.debug(f"\nTraining completed:")
    logger.debug(f"Simple RAG reward: {simple_reward:.4f}")
    logger.debug(f"Best RL-enhanced RAG reward: {best_reward:.4f}")
    logger.debug(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")
    return policy, rewards_history, actions_history, best_response


def compare_rag_approaches(
    query_text: str,
    ground_truth: str,
    vector_store: SimpleVectorStore,
    embed_func,
    mlx
) -> Tuple[str, str, float, float]:
    """Compare simple and RL-enhanced RAG."""
    logger.debug("=" * 80)
    logger.debug(f"Query: {query_text}")
    logger.debug("=" * 80)
    simple_response = basic_rag_pipeline(
        query_text, vector_store, embed_func, mlx)
    simple_similarity = calculate_reward(
        simple_response, ground_truth, embed_func)
    logger.debug("\nSimple RAG Output:")
    logger.debug("-" * 40)
    logger.debug(simple_response)
    logger.debug(f"Similarity to ground truth: {simple_similarity:.4f}")
    logger.debug("\nTraining RL-enhanced RAG model...")
    params = initialize_training_params()
    params["num_episodes"] = 5
    _, rewards_history, actions_history, best_rl_response = training_loop(
        query_text, ground_truth, vector_store, embed_func, mlx, params
    )
    if best_rl_response is None:
        query_embedding = embed_func(query_text)
        context_chunks = [chunk["text"]
                          for chunk in vector_store.search(query_embedding, top_k=5)]
        prompt = construct_prompt(query_text, context_chunks)
        best_rl_response = generate_ai_response(
            query_text,
            prompt,
            [{"text": chunk} for chunk in context_chunks],
            mlx,
            logger,
            max_tokens=512,
            temperature=1.0
        )
    rl_similarity = calculate_reward(
        best_rl_response, ground_truth, embed_func)
    logger.debug("\nRL-enhanced RAG Output:")
    logger.debug("-" * 40)
    logger.debug(best_rl_response)
    logger.debug(f"Similarity to ground truth: {rl_similarity:.4f}")
    improvement = rl_similarity - simple_similarity
    logger.debug("\nEvaluation Results:")
    logger.debug("-" * 40)
    logger.debug(
        f"Simple RAG similarity to ground truth: {simple_similarity:.4f}")
    logger.debug(
        f"RL-enhanced RAG similarity to ground truth: {rl_similarity:.4f}")
    logger.debug(f"Improvement: {improvement * 100:.2f}%")
    if len(rewards_history) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history)
        plt.title('Reward History During RL Training')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(DATA_DIR, 'reward_history.png'))
    return simple_response, best_rl_response, simple_similarity, rl_similarity


def evaluate_relevance(retrieved_chunks: List[str], ground_truth_chunks: List[str], embed_func) -> float:
    """Evaluate relevance of retrieved chunks."""
    relevance_scores = []
    for retrieved, ground_truth in zip(retrieved_chunks, ground_truth_chunks):
        relevance = np.dot(embed_func(retrieved), embed_func(ground_truth)) / (
            np.linalg.norm(embed_func(retrieved)) *
            np.linalg.norm(embed_func(ground_truth))
        )
        relevance_scores.append(relevance)
    return np.mean(relevance_scores)


def evaluate_accuracy(responses: List[str], ground_truth_responses: List[str], embed_func) -> float:
    """Evaluate accuracy of responses."""
    accuracy_scores = []
    for response, ground_truth in zip(responses, ground_truth_responses):
        accuracy = np.dot(embed_func(response), embed_func(ground_truth)) / (
            np.linalg.norm(embed_func(response)) *
            np.linalg.norm(embed_func(ground_truth))
        )
        accuracy_scores.append(accuracy)
    return np.mean(accuracy_scores)


def evaluate_response_quality(responses: List[str]) -> float:
    """Evaluate quality of responses based on length."""
    quality_scores = [min(len(response.split()) / 100, 1.0)
                      for response in responses]
    return np.mean(quality_scores)


def evaluate_rag_performance(
    queries: List[str],
    ground_truth_chunks: List[str],
    ground_truth_responses: List[str],
    vector_store: SimpleVectorStore,
    embed_func,
    mlx
) -> Dict[str, float]:
    """Evaluate RAG performance."""
    relevance_scores = []
    accuracy_scores = []
    quality_scores = []
    for query, ground_truth_chunk, ground_truth_response in zip(queries, ground_truth_chunks, ground_truth_responses):
        query_embedding = embed_func(query)
        retrieved_chunks = [chunk["text"]
                            for chunk in vector_store.search(query_embedding, top_k=5)]
        relevance = evaluate_relevance(
            retrieved_chunks, [ground_truth_chunk], embed_func)
        relevance_scores.append(relevance)
        response = basic_rag_pipeline(query, vector_store, embed_func, mlx)
        accuracy = evaluate_accuracy(
            [response], [ground_truth_response], embed_func)
        accuracy_scores.append(accuracy)
        quality = evaluate_response_quality([response])
        quality_scores.append(quality)
    return {
        "average_relevance": np.mean(relevance_scores),
        "average_accuracy": np.mean(accuracy_scores),
        "average_quality": np.mean(quality_scores)
    }


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
preprocessed_chunks = preprocess_chunks(original_chunks)
vector_store = SimpleVectorStore()
embeddings = generate_embeddings(preprocessed_chunks, embed_func, logger)
add_to_vector_store(vector_store, embeddings, preprocessed_chunks)
sample_query = "What is Quantum Computing?"
expected_answer = (
    "Quantum computing is a type of computing that utilizes principles of quantum mechanics, such as superposition, "
    "entanglement, and quantum interference, to process information. Unlike classical computers, which use bits to "
    "represent information as 0s or 1s, quantum computers use quantum bits or qubits, which can exist in multiple states "
    "simultaneously, enabling potentially exponential increases in computational power for certain problems."
)
response = basic_rag_pipeline(sample_query, vector_store, embed_func, mlx)
logger.debug("🔍 Running the Retrieval-Augmented Generation (RAG) pipeline...")
logger.debug(f"📥 Query: {sample_query}\n")
logger.debug("🤖 AI Response:")
logger.debug("-" * 50)
logger.debug(response.strip())
logger.debug("-" * 50)
logger.debug("✅ Ground Truth Answer:")
logger.debug("-" * 50)
logger.debug(expected_answer)
logger.debug("-" * 50)
simple_response, rl_response, simple_sim, rl_sim = compare_rag_approaches(
    sample_query, expected_answer, vector_store, embed_func, mlx
)
results = {
    "query": sample_query,
    "ground_truth": expected_answer,
    "simple_rag": {
        "response": simple_response,
        "similarity": float(simple_sim)
    },
    "rl_rag": {
        "response": rl_response,
        "similarity": float(rl_sim)
    },
    "improvement": float(rl_sim - simple_sim)
}
save_file(results, f"{generated_dir}/rl_rag_results.json")
logger.debug("\nResults saved to rl_rag_results.json")
logger.info("\n\n[DONE]", bright=True)
