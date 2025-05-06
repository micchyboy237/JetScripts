from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from typing import Dict, List, Tuple, Optional, Union
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pypdf

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
DATA_DIR = os.path.join(script_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def extract_text_from_pdf(pdf_path: str) -> str:
    logger.debug(f"Extracting text from {pdf_path}...")
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


def split_into_chunks(text: str, chunk_size: int = 100) -> List[str]:
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text


def preprocess_chunks(chunks: List[str]) -> List[str]:
    return [preprocess_text(chunk) for chunk in chunks]


def generate_embeddings(chunks: List[str]) -> np.ndarray:
    embeddings = embed_func(chunks)
    return np.array(embeddings)


def save_embeddings(embeddings: np.ndarray, output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(embeddings.tolist(), file)


vector_store: dict[int, dict[str, object]] = {}


def add_to_vector_store(embeddings: np.ndarray, chunks: List[str]) -> None:
    for embedding, chunk in zip(embeddings, chunks):
        vector_store[len(vector_store)] = {
            "embedding": embedding, "chunk": chunk}


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def similarity_search(query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
    similarities = []
    for key, value in vector_store.items():
        similarity = cosine_similarity(query_embedding, value["embedding"])
        similarities.append((key, similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [vector_store[key]["chunk"] for key, _ in similarities[:top_k]]


def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[str]:
    query_embedding = generate_embeddings([query_text])[0]
    relevant_chunks = similarity_search(query_embedding, top_k=top_k)
    return relevant_chunks


def construct_prompt(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)
    system_message = (
        "You are a helpful assistant. Only use the provided context to answer the question. "
        "If the context doesn't contain the information needed, say 'I don't have enough information to answer this question.'"
    )
    prompt = f"System: {system_message}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    return prompt


def generate_response(
    prompt: str,
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_tokens: int = 512,
    temperature: float = 1.0
) -> str:
    response = mlx.chat(
        [
            {"role": "user", "content": prompt}
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response["choices"][0]["message"]["content"]


def basic_rag_pipeline(query: str) -> str:
    relevant_chunks: List[str] = retrieve_relevant_chunks(query)
    prompt: str = construct_prompt(query, relevant_chunks)
    response: str = generate_response(prompt)
    return response


def define_state(
    query: str,
    context_chunks: List[str],
    rewritten_query: str = None,
    previous_responses: List[str] = None,
    previous_rewards: List[float] = None
) -> dict:
    state = {
        "original_query": query,
        "current_query": rewritten_query if rewritten_query else query,
        "context": context_chunks,
        "previous_responses": previous_responses if previous_responses else [],
        "previous_rewards": previous_rewards if previous_rewards else []
    }
    return state


def define_action_space() -> List[str]:
    actions = ["rewrite_query", "expand_context",
               "filter_context", "generate_response"]
    return actions


def calculate_reward(response: str, ground_truth: str) -> float:
    response_embedding = generate_embeddings([response])[0]
    ground_truth_embedding = generate_embeddings([ground_truth])[0]
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)
    return similarity


def rewrite_query(
    query: str,
    context_chunks: List[str],
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_tokens: int = 100,
    temperature: float = 0.3
) -> str:
    system_prompt = "You are a helpful assistant. Rewrite the query to make it more precise and contextually relevant, based on the provided context."
    context = "\n".join(context_chunks)
    user_prompt = f"Context:\n{context}\n\nOriginal Query: {query}\n\nRewritten Query:"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    rewritten_query = response["choices"][0]["message"]["content"].strip()
    return rewritten_query


def expand_context(query: str, current_chunks: List[str], top_k: int = 3) -> List[str]:
    additional_chunks = retrieve_relevant_chunks(
        query, top_k=top_k + len(current_chunks))
    new_chunks = []
    for chunk in additional_chunks:
        if chunk not in current_chunks:
            new_chunks.append(chunk)
    expanded_context = current_chunks + new_chunks[:top_k]
    return expanded_context


def filter_context(query: str, context_chunks: List[str]) -> List[str]:
    if not context_chunks:
        return []
    query_embedding = generate_embeddings([query])[0]
    chunk_embeddings = [generate_embeddings(
        [chunk])[0] for chunk in context_chunks]
    relevance_scores = []
    for chunk_embedding in chunk_embeddings:
        score = cosine_similarity(query_embedding, chunk_embedding)
        relevance_scores.append(score)
    sorted_chunks = [x for _, x in sorted(
        zip(relevance_scores, context_chunks), reverse=True)]
    filtered_chunks = sorted_chunks[:min(5, len(sorted_chunks))]
    return filtered_chunks


def policy_network(
    state: dict,
    action_space: List[str],
    epsilon: float = 0.2
) -> str:
    if np.random.random() < epsilon:
        action = np.random.choice(action_space)
    else:
        if len(state["previous_responses"]) == 0:
            action = "rewrite_query"
        elif state["previous_rewards"] and max(state["previous_rewards"]) < 0.7:
            action = "expand_context"
        elif len(state["context"]) > 5:
            action = "filter_context"
        else:
            action = "generate_response"
    return action


def rl_step(
    state: dict,
    action_space: List[str],
    ground_truth: str
) -> tuple[dict, str, float, str]:
    action: str = policy_network(state, action_space)
    response: str = None
    reward: float = 0
    if action == "rewrite_query":
        rewritten_query: str = rewrite_query(
            state["original_query"], state["context"])
        state["current_query"] = rewritten_query
        new_context: List[str] = retrieve_relevant_chunks(rewritten_query)
        state["context"] = new_context
    elif action == "expand_context":
        expanded_context: List[str] = expand_context(
            state["current_query"], state["context"])
        state["context"] = expanded_context
    elif action == "filter_context":
        filtered_context: List[str] = filter_context(
            state["current_query"], state["context"])
        state["context"] = filtered_context
    elif action == "generate_response":
        prompt: str = construct_prompt(
            state["current_query"], state["context"])
        response: str = generate_response(prompt)
        reward: float = calculate_reward(response, ground_truth)
        state["previous_responses"].append(response)
        state["previous_rewards"].append(reward)
    return state, action, reward, response


def initialize_training_params() -> Dict[str, Union[float, int]]:
    params = {
        "learning_rate": 0.01,
        "num_episodes": 100,
        "discount_factor": 0.99
    }
    return params


def update_policy(
    policy: Dict[str, Dict[str, Union[float, str]]],
    state: Dict[str, object],
    action: str,
    reward: float,
    learning_rate: float
) -> Dict[str, Dict[str, Union[float, str]]]:
    policy[state["original_query"]] = {
        "action": action,
        "reward": reward
    }
    return policy


def track_progress(
    episode: int,
    reward: float,
    rewards_history: List[float]
) -> List[float]:
    rewards_history.append(reward)
    logger.debug(f"Episode {episode}: Reward = {reward}")
    return rewards_history


def training_loop(
    query_text: str,
    ground_truth: str,
    params: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[Dict[str, Dict[str, Union[float, str]]], List[float], List[List[str]], Optional[str]]:
    if params is None:
        params = initialize_training_params()
    rewards_history: List[float] = []
    actions_history: List[List[str]] = []
    policy: Dict[str, Dict[str, Union[float, str]]] = {}
    action_space: List[str] = define_action_space()
    best_response: Optional[str] = None
    best_reward: float = -1
    simple_response: str = basic_rag_pipeline(query_text)
    simple_reward: float = calculate_reward(simple_response, ground_truth)
    logger.debug(f"Simple RAG reward: {simple_reward:.4f}")
    for episode in range(params["num_episodes"]):
        context_chunks: List[str] = retrieve_relevant_chunks(query_text)
        state: Dict[str, object] = define_state(query_text, context_chunks)
        episode_reward: float = 0
        episode_actions: List[str] = []
        for step in range(10):
            state, action, reward, response = rl_step(
                state, action_space, ground_truth)
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
    improvement: float = best_reward - simple_reward
    logger.debug(f"\nTraining completed:")
    logger.debug(f"Simple RAG reward: {simple_reward:.4f}")
    logger.debug(f"Best RL-enhanced RAG reward: {best_reward:.4f}")
    logger.debug(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")
    return policy, rewards_history, actions_history, best_response


def compare_rag_approaches(query_text: str, ground_truth: str) -> Tuple[str, str, float, float]:
    logger.debug("=" * 80)
    logger.debug(f"Query: {query_text}")
    logger.debug("=" * 80)
    simple_response: str = basic_rag_pipeline(query_text)
    simple_similarity: float = calculate_reward(simple_response, ground_truth)
    logger.debug("\nSimple RAG Output:")
    logger.debug("-" * 40)
    logger.debug(simple_response)
    logger.debug(f"Similarity to ground truth: {simple_similarity:.4f}")
    logger.debug("\nTraining RL-enhanced RAG model...")
    params: Dict[str, float | int] = initialize_training_params()
    params["num_episodes"] = 5
    _, rewards_history, actions_history, best_rl_response = training_loop(
        query_text, ground_truth, params
    )
    if best_rl_response is None:
        context_chunks: List[str] = retrieve_relevant_chunks(query_text)
        prompt: str = construct_prompt(query_text, context_chunks)
        best_rl_response: str = generate_response(prompt)
    rl_similarity: float = calculate_reward(best_rl_response, ground_truth)
    logger.debug("\nRL-enhanced RAG Output:")
    logger.debug("-" * 40)
    logger.debug(best_rl_response)
    logger.debug(f"Similarity to ground truth: {rl_similarity:.4f}")
    improvement: float = rl_similarity - simple_similarity
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


def evaluate_relevance(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
    relevance_scores: List[float] = []
    for retrieved, ground_truth in zip(retrieved_chunks, ground_truth_chunks):
        relevance: float = cosine_similarity(
            generate_embeddings([retrieved])[0],
            generate_embeddings([ground_truth])[0]
        )
        relevance_scores.append(relevance)
    return np.mean(relevance_scores)


def evaluate_accuracy(responses: List[str], ground_truth_responses: List[str]) -> float:
    accuracy_scores: List[float] = []
    for response, ground_truth in zip(responses, ground_truth_responses):
        accuracy: float = cosine_similarity(
            generate_embeddings([response])[0],
            generate_embeddings([ground_truth])[0]
        )
        accuracy_scores.append(accuracy)
    return np.mean(accuracy_scores)


def evaluate_response_quality(responses: List[str]) -> float:
    quality_scores: List[float] = []
    for response in responses:
        quality: float = len(response.split()) / 100
        quality_scores.append(min(quality, 1.0))
    return np.mean(quality_scores)


def evaluate_rag_performance(
    queries: List[str],
    ground_truth_chunks: List[str],
    ground_truth_responses: List[str]
) -> Dict[str, float]:
    relevance_scores: List[float] = []
    accuracy_scores: List[float] = []
    quality_scores: List[float] = []
    for query, ground_truth_chunk, ground_truth_response in zip(queries, ground_truth_chunks, ground_truth_responses):
        retrieved_chunks: List[str] = retrieve_relevant_chunks(query)
        relevance: float = evaluate_relevance(
            retrieved_chunks, [ground_truth_chunk])
        relevance_scores.append(relevance)
        response: str = basic_rag_pipeline(query)
        accuracy: float = evaluate_accuracy(
            [response], [ground_truth_response])
        accuracy_scores.append(accuracy)
        quality: float = evaluate_response_quality([response])
        quality_scores.append(quality)
    avg_relevance: float = np.mean(relevance_scores)
    avg_accuracy: float = np.mean(accuracy_scores)
    avg_quality: float = np.mean(quality_scores)
    return {
        "average_relevance": avg_relevance,
        "average_accuracy": avg_accuracy,
        "average_quality": avg_quality
    }


pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
text = extract_text_from_pdf(pdf_path)
chunks = split_into_chunks(text)
preprocessed_chunks = preprocess_chunks(chunks)
embeddings = generate_embeddings(preprocessed_chunks)
save_embeddings(embeddings, os.path.join(DATA_DIR, "embeddings.json"))
add_to_vector_store(embeddings, preprocessed_chunks)

sample_query = "What is Quantum Computing?"
expected_answer = "Quantum computing is a type of computing that utilizes principles of quantum mechanics, such as superposition, entanglement, and quantum interference, to process information. Unlike classical computers, which use bits to represent information as 0s or 1s, quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously, enabling potentially exponential increases in computational power for certain problems."
response = basic_rag_pipeline(sample_query)
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
    sample_query, expected_answer)

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
with open(os.path.join(DATA_DIR, 'rl_rag_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
logger.debug("\nResults saved to rl_rag_results.json")
logger.info("\n\n[DONE]", bright=True)
