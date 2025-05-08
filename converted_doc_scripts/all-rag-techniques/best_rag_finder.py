from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import faiss
import itertools
import numpy as np
import os
import pandas as pd
import pypdf
import re
import time
import warnings

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

pd.set_option('display.max_colwidth', 150)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore', category=FutureWarning)
logger.debug("Libraries imported successfully!")

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

logger.debug("--- Configuration Check --- ")
logger.debug(
    f"Models: Embed='{EMBEDDING_MODEL}', Gen='{GENERATION_MODEL}', Eval='{EVALUATION_MODEL}'")
logger.debug(f"Chunk Sizes to Test: {CHUNK_SIZES_TO_TEST}")
logger.debug(f"Overlaps to Test: {CHUNK_OVERLAPS_TO_TEST}")
logger.debug(f"Top-K Values to Test: {RETRIEVAL_TOP_K_TO_TEST}")
logger.debug(
    f"Generation Temp: {GENERATION_TEMPERATURE}, Max Tokens: {GENERATION_MAX_TOKENS}")
logger.debug("Configuration ready.")
logger.debug("-" * 25)


# Our knowledge base: A list of text documents about renewable energy
corpus_texts = [
    # Doc 0
    "Solar power uses PV panels or CSP systems. PV converts sunlight directly to electricity. CSP uses mirrors to heat fluid driving a turbine. It's clean but varies with weather/time. Storage (batteries) is key for consistency.",
    # Doc 1
    "Wind energy uses turbines in wind farms. It's sustainable with low operating costs. Wind speed varies, siting can be challenging (visual/noise). Offshore wind is stronger and more consistent.",
    "Hydropower uses moving water, often via dams spinning turbines. Reliable, large-scale power with flood control/water storage benefits. Big dams harm ecosystems and displace communities. Run-of-river is smaller, less disruptive.",  # Doc 2
    "Geothermal energy uses Earth's heat via steam/hot water for turbines. Consistent 24/7 power, small footprint. High initial drilling costs, sites are geographically limited.",  # Doc 3
    "Biomass energy from organic matter (wood, crops, waste). Burned directly or converted to biofuels. Uses waste, provides dispatchable power. Requires sustainable sourcing. Combustion releases emissions (carbon-neutral if balanced by regrowth)."  # Doc 4
]
test_query = "Compare the consistency and environmental impact of solar power versus hydropower."
true_answer_for_query = "Solar power's consistency varies with weather and time of day, requiring storage like batteries. Hydropower is generally reliable, but large dams have significant environmental impacts on ecosystems and communities, unlike solar power's primary impact being land use for panels."
logger.debug(f"Loaded {len(corpus_texts)} document into our corpus.")
logger.debug(f"Test Query: '{test_query}'")
logger.debug(
    f"Reference (True) Answer for evaluation: '{true_answer_for_query}'")
logger.debug("Input data is ready.")
logger.debug("-" * 25)


def chunk_text(text, chunk_size, chunk_overlap):
    words = text.split()
    total_words = len(words)
    chunks = []
    start_index = 0
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.debug(
            f"  Warning: Invalid chunk_size ({chunk_size}). Must be a positive integer. Returning the whole text as one chunk.")
        return [text]
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        logger.debug(
            f"  Warning: Invalid chunk_overlap ({chunk_overlap}). Must be a non-negative integer. Setting overlap to 0.")
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        adjusted_overlap = chunk_size // 3
        logger.debug(
            f"  Warning: chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Adjusting overlap to {adjusted_overlap}.")
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
                logger.debug(
                    f"  Warning: Chunking logic stuck (start={start_index}, next_start={next_start_index}). Forcing progress.")
                next_start_index = start_index + 1
        if next_start_index >= total_words:
            break
        start_index = next_start_index
    return chunks


logger.debug("Defining the 'chunk_text' function.")
sample_chunk_size = 150
sample_overlap = 30
sample_chunks = chunk_text(corpus_texts[0], sample_chunk_size, sample_overlap)
logger.debug(
    f"Test chunking on first doc (size={sample_chunk_size} words, overlap={sample_overlap} words): Created {len(sample_chunks)} chunks.")
if sample_chunks:
    logger.debug(f"First sample chunk:\n'{sample_chunks[0]}'")
logger.debug("-" * 25)


def calculate_cosine_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    try:
        embeddings = embed_func([text1, text2])
        embedding1 = np.array(embeddings[0]).reshape(1, -1)
        embedding2 = np.array(embeddings[1]).reshape(1, -1)
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
        return max(0.0, min(1.0, float(similarity_score)))
    except Exception as e:
        logger.debug(f"  Error calculating cosine similarity: {e}")
        return 0.0


logger.debug("Defining the 'calculate_cosine_similarity' function.")
test_sim = calculate_cosine_similarity("apple", "orange")
logger.debug(
    f"Testing similarity function: Similarity between 'apple' and 'orange' = {test_sim:.2f}")
logger.debug("-" * 25)

all_results = []
last_chunk_size = -1
last_overlap = -1
current_index = None
current_chunks = []
current_embeddings = None

logger.debug("=== Starting RAG Experiment Loop ===\n")
param_combinations = list(itertools.product(
    CHUNK_SIZES_TO_TEST,
    CHUNK_OVERLAPS_TO_TEST,
    RETRIEVAL_TOP_K_TO_TEST
))
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
            for doc_index, doc in enumerate(corpus_texts):
                doc_chunks = chunk_text(doc, chunk_size, chunk_overlap)
                if not doc_chunks:
                    logger.debug(
                        f"  Warning: No chunks created for document {doc_index} with size={chunk_size}, overlap={chunk_overlap}. Skipping document.")
                    continue
                temp_chunks.extend(doc_chunks)
            current_chunks = temp_chunks
            if not current_chunks:
                raise ValueError(
                    "No chunks were created for the current configuration.")
        except Exception as e:
            logger.debug(
                f"    ERROR during chunking for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this configuration.")
            last_chunk_size, last_overlap = -1, -1
            continue
        try:
            current_embeddings = np.array(embed_func(current_chunks))
            if current_embeddings.ndim != 2 or current_embeddings.shape[0] != len(current_chunks):
                raise ValueError(
                    f"Embeddings array shape mismatch. Expected ({len(current_chunks)}, dim), Got {current_embeddings.shape}")
        except Exception as e:
            logger.debug(
                f"    ERROR generating embeddings for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this chunk config.")
            last_chunk_size, last_overlap = -1, -1
            current_chunks = []
            current_embeddings = None
            continue
        try:
            embedding_dim = current_embeddings.shape[1]
            current_index = faiss.IndexFlatL2(embedding_dim)
            current_index.add(current_embeddings.astype('float32'))
            if current_index.ntotal == 0:
                raise ValueError(
                    "FAISS index is empty after adding vectors. No vectors were added.")
        except Exception as e:
            logger.debug(
                f"    ERROR building FAISS index for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this chunk config.")
            last_chunk_size, last_overlap = -1, -1
            current_index = None
            current_embeddings = None
            current_chunks = []
            continue
    if current_index is None or not current_chunks:
        logger.debug(
            f"    WARNING: Index or chunks not available for Size={chunk_size}, Overlap={chunk_overlap}. Skipping Top-K={top_k} test.")
        continue

    def run_and_evaluate(strategy_name, query_to_use, k_retrieve, use_simulated_rerank=False):
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
            k_for_search = k_retrieve
            if use_simulated_rerank:
                k_for_search = k_retrieve * RERANK_RETRIEVAL_MULTIPLIER
            query_embedding = np.array(embed_func(
                [query_to_use])[0]).astype('float32')
            query_vector = query_embedding.reshape(1, -1)
            actual_k = min(k_for_search, current_index.ntotal)
            if actual_k == 0:
                raise ValueError(
                    "Index is empty or k_for_search is zero, cannot search.")
            distances, indices = current_index.search(query_vector, actual_k)
            retrieved_indices_all = indices[0]
            valid_indices = retrieved_indices_all[retrieved_indices_all != -1].tolist()
            if use_simulated_rerank:
                final_indices = valid_indices[:k_retrieve]
            else:
                final_indices = valid_indices
            result['retrieved_indices'] = final_indices
            retrieved_chunks = [current_chunks[i] for i in final_indices]
            if not retrieved_chunks:
                logger.debug(
                    f"      Warning: No relevant chunks found for {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}). Setting answer to indicate this.")
                result['answer'] = "No relevant context found in the documents based on the query."
            else:
                context_str = "\n\n".join(retrieved_chunks)
                sys_prompt_gen = "You are a helpful AI assistant. Answer the user's query based strictly on the provided context. If the context doesn't contain the answer, state that clearly. Be concise."
                user_prompt_gen = f"Context:\n------\n{context_str}\n------\n\nQuery: {test_query}\n\nAnswer:"
                gen_response = mlx.chat(
                    [
                        {"role": "system", "content": sys_prompt_gen},
                        {"role": "user", "content": user_prompt_gen}
                    ],
                    model=GENERATION_MODEL,
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=GENERATION_MAX_TOKENS
                )
                generated_answer = gen_response["choices"][0]["message"]["content"].strip(
                )
                result['answer'] = generated_answer
                faithfulness_prompt = "Rate the faithfulness of the response to the provided true answer on a scale from 0 to 1, where 1 is completely faithful. Response: {response}\nTrue Answer: {true_answer}"
                prompt_f = faithfulness_prompt.format(
                    response=generated_answer, true_answer=true_answer_for_query)
                try:
                    resp_f = mlx.chat(
                        [{"role": "user", "content": prompt_f}],
                        model=EVALUATION_MODEL,
                        temperature=0.0,
                        max_tokens=10
                    )
                    result['faithfulness'] = max(
                        0.0, min(1.0, float(resp_f["choices"][0]["message"]["content"].strip())))
                except Exception as eval_e:
                    logger.debug(
                        f"      Warning: Faithfulness score parsing error for {strategy_name} - {eval_e}. Score set to 0.0")
                    result['faithfulness'] = 0.0
                relevancy_prompt = "Rate the relevancy of the response to the query on a scale from 0 to 1, where 1 is highly relevant. Query: {question}\nResponse: {response}"
                prompt_r = relevancy_prompt.format(
                    question=test_query, response=generated_answer)
                try:
                    resp_r = mlx.chat(
                        [{"role": "user", "content": prompt_r}],
                        model=EVALUATION_MODEL,
                        temperature=0.0,
                        max_tokens=10
                    )
                    result['relevancy'] = max(
                        0.0, min(1.0, float(resp_r["choices"][0]["message"]["content"].strip())))
                except Exception as eval_e:
                    logger.debug(
                        f"      Warning: Relevancy score parsing error for {strategy_name} - {eval_e}. Score set to 0.0")
                    result['relevancy'] = 0.0
                result['similarity_score'] = calculate_cosine_similarity(
                    generated_answer,
                    true_answer_for_query
                )
                result['avg_score'] = (
                    result['faithfulness'] + result['relevancy'] + result['similarity_score']) / 3.0
        except Exception as e:
            error_message = f"ERROR during {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}): {str(e)[:200]}..."
            logger.debug(f"    {error_message}")
            result['answer'] = error_message
            result['faithfulness'] = 0.0
            result['relevancy'] = 0.0
            result['similarity_score'] = 0.0
            result['avg_score'] = 0.0
        run_end_time = time.time()
        result['time_sec'] = run_end_time - run_start_time
        logger.debug(
            f"    Finished: {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}). AvgScore={result['avg_score']:.2f}, Time={result['time_sec']:.2f}s")
        return result
    result_simple = run_and_evaluate("Simple RAG", test_query, top_k)
    all_results.append(result_simple)
    rewritten_q = test_query
    try:
        sys_prompt_rw = "You are an expert query optimizer. Rewrite the user's query to be ideal for vector database retrieval. Focus on key entities, concepts, and relationships. Remove conversational fluff. Output ONLY the rewritten query text."
        user_prompt_rw = f"Original Query: {test_query}\n\nRewritten Query:"
        resp_rw = mlx.chat(
            [
                {"role": "system", "content": sys_prompt_rw},
                {"role": "user", "content": user_prompt_rw}
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
        rewritten_q = test_query
    result_rewrite = run_and_evaluate("Query Rewrite RAG", rewritten_q, top_k)
    all_results.append(result_rewrite)
    result_rerank = run_and_evaluate(
        "Rerank RAG (Simulated)", test_query, top_k, use_simulated_rerank=True)
    all_results.append(result_rerank)
logger.debug("\n=== RAG Experiment Loop Finished ===")
logger.debug("-" * 25)

logger.debug("--- Analyzing Experiment Results ---")
if not all_results:
    logger.debug(
        "No results were generated during the experiment. Cannot perform analysis.")
else:
    results_df = pd.DataFrame(all_results)
    logger.debug(f"Total results collected: {len(results_df)}")
    results_df_sorted = results_df.sort_values(
        by='avg_score', ascending=False).reset_index(drop=True)
    logger.debug(
        "\n--- Top 10 Performing Configurations (Sorted by Average Score) ---")
    display_cols = [
        'chunk_size', 'overlap', 'top_k', 'strategy',
        'avg_score', 'faithfulness', 'relevancy', 'similarity_score',
        'time_sec',
        'answer'
    ]
    display_cols = [
        col for col in display_cols if col in results_df_sorted.columns]
    logger.debug("\n" + results_df_sorted[display_cols].head(10).to_string())
    logger.debug("\n--- Best Configuration Summary ---")
    if not results_df_sorted.empty:
        best_run = results_df_sorted.iloc[0]
        logger.debug(f"Chunk Size: {best_run.get('chunk_size', 'N/A')} words")
        logger.debug(f"Overlap: {best_run.get('overlap', 'N/A')} words")
        logger.debug(f"Top-K Retrieved: {best_run.get('top_k', 'N/A')} chunks")
        logger.debug(f"Strategy: {best_run.get('strategy', 'N/A')}")
        avg_score = best_run.get('avg_score', 0.0)
        faithfulness = best_run.get('faithfulness', 0.0)
        relevancy = best_run.get('relevancy', 0.0)
        similarity = best_run.get('similarity_score', 0.0)
        time_sec = best_run.get('time_sec', 0.0)
        best_answer = best_run.get('answer', 'N/A')
        logger.debug(f"---> Average Score (Faith+Rel+Sim): {avg_score:.3f}")
        logger.debug(
            f"      (Faithfulness: {faithfulness:.3f}, Relevancy: {relevancy:.3f}, Similarity: {similarity:.3f})")
        logger.debug(f"Time Taken: {time_sec:.2f} seconds")
        logger.debug(f"\nBest Answer Generated:")
        logger.debug(best_answer)
    else:
        logger.debug(
            "Could not determine the best configuration (no valid results found).")
logger.debug("\n--- Analysis Complete --- ")

logger.info("\n\n[DONE]", bright=True)
