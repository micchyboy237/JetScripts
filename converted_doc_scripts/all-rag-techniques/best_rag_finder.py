import faiss
import itertools
import numpy as np
import os
import pandas as pd
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    generate_ai_response, load_json_data, DATA_DIR, DOCS_PATH
)

pd.set_option('display.max_colwidth', 150)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore', category=FutureWarning)

EMBEDDING_MODEL = "mxbai-embed-large"
GENERATION_MODEL = "llama-3.2-1b-instruct-4bit"
EVALUATION_MODEL = "llama-3.2-1b-instruct-4bit"
GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_TOKENS = 400
GENERATION_TOP_P = 0.9
CHUNK_SIZES_TO_TEST = [150, 250]
CHUNK_OVERLAPS_TO_TEST = [30, 50]
RETRIEVAL_TOP_K_TO_TEST = [3, 5]
RERANK_RETRIEVAL_MULTIPLIER = 3


def chunk_text(text, chunk_size, chunk_overlap):
    """Chunk text into overlapping segments."""
    words = text.split()
    total_words = len(words)
    chunks = []
    start_index = 0
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.debug(
            f"  Warning: Invalid chunk_size ({chunk_size}). Returning whole text.")
        return [text]
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        logger.debug(
            f"  Warning: Invalid chunk_overlap ({chunk_overlap}). Setting to 0.")
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        adjusted_overlap = chunk_size // 3
        logger.debug(
            f"  Warning: chunk_overlap ({chunk_overlap}) >= chunk_size. Adjusting to {adjusted_overlap}.")
        chunk_overlap = adjusted_overlap
    while start_index < total_words:
        end_index = min(start_index + chunk_size, total_words)
        current_chunk_text = " ".join(words[start_index:end_index])
        chunks.append(current_chunk_text)
        next_start_index = start_index + chunk_size - chunk_overlap
        if next_start_index <= start_index:
            if end_index == total_words:
                break
            else:
                logger.debug(f"  Warning: Chunking stuck. Forcing progress.")
                next_start_index = start_index + 1
        if next_start_index >= total_words:
            break
        start_index = next_start_index
    return chunks


def run_and_evaluate(strategy_name, query_to_use, k_retrieve, current_index, current_chunks, mlx, embed_func, use_simulated_rerank=False):
    """Run and evaluate RAG strategy."""
    run_start_time = time.time()
    result = {
        'chunk_size': chunk_size, 'overlap': chunk_overlap, 'top_k': k_retrieve,
        'strategy': strategy_name,
        'retrieved_indices': [], 'rewritten_query': None, 'answer': 'Error: Execution Failed',
        'faithfulness': 0.0, 'relevancy': 0.0, 'similarity_score': 0.0, 'avg_score': 0.0,
        'time_sec': 0.0
    }
    if strategy_name == "Query Rewrite RAG":
        result['rewritten_query'] = query_to_use
    try:
        k_for_search = k_retrieve * \
            RERANK_RETRIEVAL_MULTIPLIER if use_simulated_rerank else k_retrieve
        query_embedding = np.array(embed_func(query_to_use)).astype('float32')
        query_vector = query_embedding.reshape(1, -1)
        actual_k = min(k_for_search, current_index.ntotal)
        if actual_k == 0:
            raise ValueError("Index is empty or k_for_search is zero.")
        distances, indices = current_index.search(query_vector, actual_k)
        retrieved_indices_all = indices[0]
        valid_indices = retrieved_indices_all[retrieved_indices_all != -1].tolist()
        final_indices = valid_indices[:k_retrieve] if use_simulated_rerank else valid_indices
        result['retrieved_indices'] = final_indices
        retrieved_chunks = [current_chunks[i] for i in final_indices]
        if not retrieved_chunks:
            logger.debug(
                f"      Warning: No chunks found for {strategy_name}.")
            result['answer'] = "No relevant context found."
        else:
            context_str = "\n\n".join(retrieved_chunks)
            sys_prompt_gen = "You are a helpful AI assistant. Answer the query based strictly on the provided context. If the context doesn't contain the answer, state that clearly. Be concise."
            response = generate_ai_response(
                test_query,
                f"System: {sys_prompt_gen}\n\nContext:\n------\n{context_str}\n------\n\nQuery: {test_query}\n\nAnswer:",
                [{"text": chunk} for chunk in retrieved_chunks],
                mlx,
                logger,
                model=GENERATION_MODEL,
                temperature=GENERATION_TEMPERATURE,
                max_tokens=GENERATION_MAX_TOKENS
            )
            result['answer'] = response
            faithfulness_prompt = "Rate the faithfulness of the response to the provided true answer on a scale from 0 to 1, where 1 is completely faithful. Response: {response}\nTrue Answer: {true_answer}"
            try:
                faithfulness_resp = mlx.chat(
                    [{"role": "user", "content": faithfulness_prompt.format(
                        response=response, true_answer=true_answer_for_query)}],
                    model=EVALUATION_MODEL,
                    temperature=0.0,
                    max_tokens=10
                )
                result['faithfulness'] = max(0.0, min(1.0, float(
                    faithfulness_resp["choices"][0]["message"]["content"].strip())))
            except Exception as e:
                logger.debug(
                    f"      Warning: Faithfulness score error: {e}. Score set to 0.0")
                result['faithfulness'] = 0.0
            relevancy_prompt = "Rate the relevancy of the response to the query on a scale from 0 to 1, where 1 is highly relevant. Query: {question}\nResponse: {response}"
            try:
                relevancy_resp = mlx.chat(
                    [{"role": "user", "content": relevancy_prompt.format(
                        question=test_query, response=response)}],
                    model=EVALUATION_MODEL,
                    temperature=0.0,
                    max_tokens=10
                )
                result['relevancy'] = max(0.0, min(1.0, float(
                    relevancy_resp["choices"][0]["message"]["content"].strip())))
            except Exception as e:
                logger.debug(
                    f"      Warning: Relevancy score error: {e}. Score set to 0.0")
                result['relevancy'] = 0.0
            embeddings = generate_embeddings(
                [response, true_answer_for_query], embed_func, logger)
            result['similarity_score'] = cosine_similarity(
                [embeddings[0]], [embeddings[1]])[0][0]
            result['avg_score'] = (
                result['faithfulness'] + result['relevancy'] + result['similarity_score']) / 3.0
    except Exception as e:
        logger.debug(
            f"    ERROR during {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}): {str(e)[:200]}...")
        result['answer'] = f"ERROR: {str(e)[:200]}..."
    result['time_sec'] = time.time() - run_start_time
    logger.debug(
        f"    Finished: {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}). AvgScore={result['avg_score']:.2f}, Time={result['time_sec']:.2f}s")
    return result


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
test_query = "Compare the consistency and environmental impact of solar power versus hydropower."
true_answer_for_query = (
    "Solar power's consistency varies with weather and time of day, requiring storage like batteries. "
    "Hydropower is generally reliable, but large dams have significant environmental impacts on ecosystems and communities, "
    "unlike solar power's primary impact being land use for panels."
)
logger.debug(f"Test Query: '{test_query}'")
logger.debug(f"Reference Answer: '{true_answer_for_query}'")

all_results = []
last_chunk_size = -1
last_overlap = -1
current_index = None
current_chunks = []
current_embeddings = None

param_combinations = list(itertools.product(
    CHUNK_SIZES_TO_TEST, CHUNK_OVERLAPS_TO_TEST, RETRIEVAL_TOP_K_TO_TEST))
logger.debug(
    f"Total parameter combinations to test: {len(param_combinations)}")

for chunk_size, chunk_overlap, top_k in tqdm(param_combinations, desc="Testing Configurations"):
    if chunk_size != last_chunk_size or chunk_overlap != last_overlap:
        last_chunk_size, last_overlap = chunk_size, chunk_overlap
        current_index = None
        current_chunks = []
        current_embeddings = None
        try:
            temp_chunks = []
            for doc in original_chunks:
                doc_chunks = chunk_text(doc, chunk_size, chunk_overlap)
                if not doc_chunks:
                    logger.debug(
                        f"  Warning: No chunks created for document with size={chunk_size}, overlap={chunk_overlap}.")
                    continue
                temp_chunks.extend(doc_chunks)
            current_chunks = temp_chunks
            if not current_chunks:
                raise ValueError(
                    "No chunks created for current configuration.")
        except Exception as e:
            logger.debug(
                f"    ERROR during chunking for Size={chunk_size}, Overlap={chunk_overlap}: {e}.")
            last_chunk_size, last_overlap = -1, -1
            continue
        try:
            current_embeddings = np.array(
                generate_embeddings(current_chunks, embed_func, logger))
            if current_embeddings.ndim != 2 or current_embeddings.shape[0] != len(current_chunks):
                raise ValueError(
                    f"Embeddings shape mismatch. Expected ({len(current_chunks)}, dim), Got {current_embeddings.shape}")
        except Exception as e:
            logger.debug(
                f"    ERROR generating embeddings for Size={chunk_size}, Overlap={chunk_overlap}: {e}.")
            last_chunk_size, last_overlap = -1, -1
            current_chunks = []
            current_embeddings = None
            continue
        try:
            embedding_dim = current_embeddings.shape[1]
            current_index = faiss.IndexFlatL2(embedding_dim)
            current_index.add(current_embeddings.astype('float32'))
            if current_index.ntotal == 0:
                raise ValueError("FAISS index is empty.")
        except Exception as e:
            logger.debug(
                f"    ERROR building FAISS index for Size={chunk_size}, Overlap={chunk_overlap}: {e}.")
            last_chunk_size, last_overlap = -1, -1
            current_index = None
            current_embeddings = None
            current_chunks = []
            continue
    if current_index is None or not current_chunks:
        logger.debug(
            f"    WARNING: Index or chunks not available for Size={chunk_size}, Overlap={chunk_overlap}. Skipping Top-K={top_k}.")
        continue
    result_simple = run_and_evaluate(
        "Simple RAG", test_query, top_k, current_index, current_chunks, mlx, embed_func)
    all_results.append(result_simple)
    rewritten_q = test_query
    try:
        sys_prompt_rw = "You are an expert query optimizer. Rewrite the query to be ideal for vector database retrieval. Focus on key entities, concepts, and relationships. Remove conversational fluff. Output ONLY the rewritten query text."
        resp_rw = mlx.chat(
            [
                {"role": "system", "content": sys_prompt_rw},
                {"role": "user", "content": f"Original Query: {test_query}\n\nRewritten Query:"}
            ],
            model=GENERATION_MODEL,
            temperature=0.1,
            max_tokens=100
        )
        candidate_q = resp_rw["choices"][0]["message"]["content"].strip()
        candidate_q = re.sub(r'^(rewritten query:|query:)\s*',
                             '', candidate_q, flags=re.IGNORECASE).strip('"')
        if candidate_q and len(candidate_q) > 5 and candidate_q.lower() != test_query.lower():
            rewritten_q = candidate_q
    except Exception as e:
        logger.debug(
            f"    Warning: Error during query rewrite: {e}. Using original query.")
    result_rewrite = run_and_evaluate(
        "Query Rewrite RAG", rewritten_q, top_k, current_index, current_chunks, mlx, embed_func)
    all_results.append(result_rewrite)
    result_rerank = run_and_evaluate("Rerank RAG (Simulated)", test_query, top_k,
                                     current_index, current_chunks, mlx, embed_func, use_simulated_rerank=True)
    all_results.append(result_rerank)

logger.debug("\n=== RAG Experiment Loop Finished ===")
logger.debug("--- Analyzing Experiment Results ---")
if not all_results:
    logger.debug("No results generated during the experiment.")
else:
    results_df = pd.DataFrame(all_results)
    logger.debug(f"Total results collected: {len(results_df)}")
    results_df_sorted = results_df.sort_values(
        by='avg_score', ascending=False).reset_index(drop=True)
    logger.debug("\n--- Top 10 Performing Configurations ---")
    display_cols = ['chunk_size', 'overlap', 'top_k', 'strategy', 'avg_score',
                    'faithfulness', 'relevancy', 'similarity_score', 'time_sec', 'answer']
    logger.debug("\n" + results_df_sorted[display_cols].head(10).to_string())
    logger.debug("\n--- Best Configuration Summary ---")
    if not results_df_sorted.empty:
        best_run = results_df_sorted.iloc[0]
        logger.debug(f"Chunk Size: {best_run.get('chunk_size', 'N/A')} words")
        logger.debug(f"Overlap: {best_run.get('overlap', 'N/A')} words")
        logger.debug(f"Top-K Retrieved: {best_run.get('top_k', 'N/A')} chunks")
        logger.debug(f"Strategy: {best_run.get('strategy', 'N/A')}")
        logger.debug(
            f"---> Average Score: {best_run.get('avg_score', 0.0):.3f}")
        logger.debug(
            f"      (Faithfulness: {best_run.get('faithfulness', 0.0):.3f}, Relevancy: {best_run.get('relevancy', 0.0):.3f}, Similarity: {best_run.get('similarity_score', 0.0):.3f})")
        logger.debug(
            f"Time Taken: {best_run.get('time_sec', 0.0):.2f} seconds")
        logger.debug(
            f"\nBest Answer Generated:\n{best_run.get('answer', 'N/A')}")
    else:
        logger.debug("Could not determine best configuration.")
logger.info("\n\n[DONE]", bright=True)
