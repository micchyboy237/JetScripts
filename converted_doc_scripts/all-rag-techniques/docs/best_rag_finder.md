# Learning RAG: Testing Configurations Step-by-Step
## An Educational End-to-End Pipeline with Enhanced Evaluation

This notebook is designed as a learning project to understand how different settings impact Retrieval-Augmented Generation (RAG) systems. We'll build and test a pipeline step-by-step using the **Nebius AI API**.

**What we'll learn:**
*   How text chunking (`chunk_size`, `chunk_overlap`) affects what the RAG system retrieves.
*   How the number of retrieved documents (`top_k`) influences the context provided to the LLM.
*   The difference between three common RAG strategies (Simple, Query Rewrite, Rerank).
*   How to use an LLM (like Nebius AI) to automatically evaluate the quality of generated answers using multiple metrics: **Faithfulness**, **Relevancy**, and **Semantic Similarity** to a ground truth answer.
*   How to combine these metrics into an average score for easier comparison.

We'll focus on understanding *why* we perform each step and observing the outcomes clearly, with detailed explanations and commented code.

### Table of Contents
1.  **Setup: Installing Libraries**: Get the necessary tools.
2.  **Setup: Importing Libraries**: Bring the tools into our workspace.
3.  **Configuration: Setting Up Our Experiment**: Define API details, models, evaluation prompts, and parameters to test.
4.  **Input Data: The Knowledge Source & Our Question**: Define the documents the RAG system will learn from and the question we'll ask.
5.  **Core Component: Text Chunking Function**: Create a function to break documents into smaller pieces.
6.  **Core Component: Connecting to Nebius AI**: Establish the connection to use Nebius models.
7.  **Core Component: Cosine Similarity Function**: Create a function to measure semantic similarity between texts.
8.  **The Experiment: Iterating Through Configurations**: The main loop where we test different settings.
    *   8.1 Processing a Chunking Configuration (Chunk, Embed, Index)
    *   8.2 Testing RAG Strategies for a `top_k` Value
    *   8.3 Running & Evaluating a Single RAG Strategy (including Similarity)
9.  **Analysis: Reviewing the Results**: Use Pandas to organize and display the results.
10. **Conclusion: What Did We Learn?**: Reflect on the findings and potential next steps.

### 1. Setup: Installing Libraries

First, we need to install the Python packages required for this notebook.
- `openai`: Interacts with the Nebius API (which uses an OpenAI-compatible interface).
- `pandas`: For creating and managing data tables (DataFrames).
- `numpy`: For numerical operations, especially with vectors (embeddings).
- `faiss-cpu`: For efficient similarity search on vectors (the retrieval part).
- `ipywidgets`, `tqdm`: For displaying progress bars in Jupyter.
- `scikit-learn`: For calculating cosine similarity.

```python

```

**Remember!** After the installation finishes, you might need to **Restart the Kernel** (or Runtime) for Jupyter/Colab to recognize the newly installed packages. Look for this option in the menu (e.g., 'Kernel' -> 'Restart Kernel...' or 'Runtime' -> 'Restart Runtime').

### 2. Setup: Importing Libraries

With the libraries installed, we import them into our Python environment to make their functions available.

```python
import os
import time
import re
import warnings
import itertools
import getpass

import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity


pd.set_option('display.max_colwidth', 150)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore', category=FutureWarning)

print("Libraries imported successfully!")
```

```output
Libraries imported successfully!
```

### 3. Configuration: Setting Up Our Experiment

Here, we define all the settings and parameters for our experiment directly as Python variables. This makes it easy to see and modify the configuration in one place.

**Key Configuration Areas:**
*   **Nebius API Details:** Credentials and model identifiers for connecting to Nebius AI.
*   **LLM Settings:** Parameters controlling the behavior of the language model during answer generation (e.g., `temperature` for creativity).
*   **Evaluation Prompts:** The specific instructions (prompts) given to the LLM when it acts as an evaluator for Faithfulness and Relevancy.
*   **Tuning Parameters:** The different values for chunk size, overlap, and retrieval `top_k` that we want to systematically test.
*   **Reranking Setting:** Configuration for the simulated reranking strategy.

```python



NEBIUS_API_KEY = os.getenv('NEBIUS_API_KEY', None)
if NEBIUS_API_KEY is None:
    print("Warning: NEBIUS_API_KEY not set. Please set it in your environment variables or provide it directly in the code.")
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
NEBIUS_EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"
NEBIUS_GENERATION_MODEL = "deepseek-ai/DeepSeek-V3"
NEBIUS_EVALUATION_MODEL = "deepseek-ai/DeepSeek-V3"


GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_TOKENS = 400
GENERATION_TOP_P = 0.9



FAITHFULNESS_PROMPT = """
System: You are an objective evaluator. Evaluate the faithfulness of the AI Response compared to the True Answer, considering only the information present in the True Answer as the ground truth.
Faithfulness measures how accurately the AI response reflects the information in the True Answer, without adding unsupported facts or contradicting it.
Score STRICTLY using a float between 0.0 and 1.0, based on this scale:
- 0.0: Completely unfaithful, contradicts or fabricates information.
- 0.1-0.4: Low faithfulness with significant inaccuracies or unsupported claims.
- 0.5-0.6: Partially faithful but with noticeable inaccuracies or omissions.
- 0.7-0.8: Mostly faithful with only minor inaccuracies or phrasing differences.
- 0.9: Very faithful, slight wording differences but semantically aligned.
- 1.0: Completely faithful, accurately reflects the True Answer.
Respond ONLY with the numerical score.

User:
Query: {question}
AI Response: {response}
True Answer: {true_answer}
Score:"""


RELEVANCY_PROMPT = """
System: You are an objective evaluator. Evaluate the relevance of the AI Response to the specific User Query.
Relevancy measures how well the response directly answers the user's question, avoiding unnecessary or off-topic information.
Score STRICTLY using a float between 0.0 and 1.0, based on this scale:
- 0.0: Not relevant at all.
- 0.1-0.4: Low relevance, addresses a different topic or misses the core question.
- 0.5-0.6: Partially relevant, answers only a part of the query or is tangentially related.
- 0.7-0.8: Mostly relevant, addresses the main aspects of the query but might include minor irrelevant details.
- 0.9: Highly relevant, directly answers the query with minimal extra information.
- 1.0: Completely relevant, directly and fully answers the exact question asked.
Respond ONLY with the numerical score.

User:
Query: {question}
AI Response: {response}
Score:"""


CHUNK_SIZES_TO_TEST = [150, 250]
CHUNK_OVERLAPS_TO_TEST = [30, 50]
RETRIEVAL_TOP_K_TO_TEST = [3, 5]


RERANK_RETRIEVAL_MULTIPLIER = 3


print("--- Configuration Check --- ")
print(f"Attempting to load Nebius API Key from environment variable 'NEBIUS_API_KEY'...")
if not NEBIUS_API_KEY:
    print("Nebius API Key not found in environment variables.")

    NEBIUS_API_KEY = getpass.getpass("Please enter your Nebius API Key: ")
else:
    print("Nebius API Key loaded successfully from environment variable.")


print(f"Models: Embed='{NEBIUS_EMBEDDING_MODEL}', Gen='{NEBIUS_GENERATION_MODEL}', Eval='{NEBIUS_EVALUATION_MODEL}'")
print(f"Chunk Sizes to Test: {CHUNK_SIZES_TO_TEST}")
print(f"Overlaps to Test: {CHUNK_OVERLAPS_TO_TEST}")
print(f"Top-K Values to Test: {RETRIEVAL_TOP_K_TO_TEST}")
print(f"Generation Temp: {GENERATION_TEMPERATURE}, Max Tokens: {GENERATION_MAX_TOKENS}")
print("Configuration ready.")
print("-" * 25)
```

```output
--- Configuration Check ---
Attempting to load Nebius API Key from environment variable 'NEBIUS_API_KEY'...
Nebius API Key loaded successfully from environment variable.
Models: Embed='BAAI/bge-multilingual-gemma2', Gen='deepseek-ai/DeepSeek-V3', Eval='deepseek-ai/DeepSeek-V3'
Chunk Sizes to Test: [150, 250]
Overlaps to Test: [30, 50]
Top-K Values to Test: [3, 5]
Generation Temp: 0.1, Max Tokens: 400
Configuration ready.
-------------------------
```

### 4. Input Data: The Knowledge Source & Our Question

Every RAG system needs a knowledge base to draw information from. Here, we define:
*   `corpus_texts`: A list of strings, where each string is a document containing information (in this case, about renewable energy sources).
*   `test_query`: The specific question we want the RAG system to answer using the `corpus_texts`.
*   `true_answer_for_query`: A carefully crafted 'ground truth' answer based *only* on the information available in `corpus_texts`. This is essential for evaluating Faithfulness and Semantic Similarity accurately.

```python

corpus_texts = [
    "Solar power uses PV panels or CSP systems. PV converts sunlight directly to electricity. CSP uses mirrors to heat fluid driving a turbine. It's clean but varies with weather/time. Storage (batteries) is key for consistency.",
    "Wind energy uses turbines in wind farms. It's sustainable with low operating costs. Wind speed varies, siting can be challenging (visual/noise). Offshore wind is stronger and more consistent.",
    "Hydropower uses moving water, often via dams spinning turbines. Reliable, large-scale power with flood control/water storage benefits. Big dams harm ecosystems and displace communities. Run-of-river is smaller, less disruptive.",
    "Geothermal energy uses Earth's heat via steam/hot water for turbines. Consistent 24/7 power, small footprint. High initial drilling costs, sites are geographically limited.",
    "Biomass energy from organic matter (wood, crops, waste). Burned directly or converted to biofuels. Uses waste, provides dispatchable power. Requires sustainable sourcing. Combustion releases emissions (carbon-neutral if balanced by regrowth)."
]


test_query = "Compare the consistency and environmental impact of solar power versus hydropower."



true_answer_for_query = "Solar power's consistency varies with weather and time of day, requiring storage like batteries. Hydropower is generally reliable, but large dams have significant environmental impacts on ecosystems and communities, unlike solar power's primary impact being land use for panels."

print(f"Loaded {len(corpus_texts)} documents into our corpus.")
print(f"Test Query: '{test_query}'")
print(f"Reference (True) Answer for evaluation: '{true_answer_for_query}'")
print("Input data is ready.")
print("-" * 25)
```

```output
Loaded 5 documents into our corpus.
Test Query: 'Compare the consistency and environmental impact of solar power versus hydropower.'
Reference (True) Answer for evaluation: 'Solar power's consistency varies with weather and time of day, requiring storage like batteries. Hydropower is generally reliable, but large dams have significant environmental impacts on ecosystems and communities, unlike solar power's primary impact being land use for panels.'
Input data is ready.
-------------------------
```

### 5. Core Component: Text Chunking Function

LLMs and embedding models have limits on the amount of text they can process at once. Furthermore, retrieval works best when searching over smaller, focused pieces of text rather than entire large documents.

**Chunking** is the process of splitting large documents into smaller, potentially overlapping, segments.

- **`chunk_size`**: Determines the approximate size (here, in words) of each chunk.
- **`chunk_overlap`**: Specifies how many words from the end of one chunk should also be included at the beginning of the next chunk. This helps prevent relevant information from being lost if it spans across the boundary between two chunks.

We define a function `chunk_text` to perform this splitting based on word counts.

```python
def chunk_text(text, chunk_size, chunk_overlap):
    """Splits a single text document into overlapping chunks based on word count.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The target number of words per chunk.
        chunk_overlap (int): The number of words to overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    words = text.split()
    total_words = len(words)
    chunks = []
    start_index = 0



    if not isinstance(chunk_size, int) or chunk_size <= 0:
        print(f"  Warning: Invalid chunk_size ({chunk_size}). Must be a positive integer. Returning the whole text as one chunk.")
        return [text]

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        print(f"  Warning: Invalid chunk_overlap ({chunk_overlap}). Must be a non-negative integer. Setting overlap to 0.")
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:


        adjusted_overlap = chunk_size // 3
        print(f"  Warning: chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Adjusting overlap to {adjusted_overlap}.")
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

                print(f"  Warning: Chunking logic stuck (start={start_index}, next_start={next_start_index}). Forcing progress.")
                next_start_index = start_index + 1


        if next_start_index >= total_words:
            break


        start_index = next_start_index

    return chunks



print("Defining the 'chunk_text' function.")
sample_chunk_size = 150
sample_overlap = 30
sample_chunks = chunk_text(corpus_texts[0], sample_chunk_size, sample_overlap)
print(f"Test chunking on first doc (size={sample_chunk_size} words, overlap={sample_overlap} words): Created {len(sample_chunks)} chunks.")
if sample_chunks:
    print(f"First sample chunk:\n'{sample_chunks[0]}'")
print("-" * 25)
```

```output
Defining the 'chunk_text' function.
Test chunking on first doc (size=150 words, overlap=30 words): Created 1 chunks.
First sample chunk:
'Solar power uses PV panels or CSP systems. PV converts sunlight directly to electricity. CSP uses mirrors to heat fluid driving a turbine. It's clean but varies with weather/time. Storage (batteries) is key for consistency.'
-------------------------
```

### 6. Core Component: Connecting to Nebius AI

To use the Nebius AI models (for embedding, generation, evaluation), we need to establish a connection to their API. We use the `openai` Python library, which provides a convenient way to interact with OpenAI-compatible APIs like Nebius.

We instantiate an `OpenAI` client object, providing our API key and the specific Nebius API endpoint URL.

```python
client = None

print("Attempting to initialize the Nebius AI client...")
try:

    if not NEBIUS_API_KEY:
        raise ValueError("Nebius API Key is missing. Cannot initialize client.")


    client = OpenAI(
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL
    )









    print("Nebius AI client initialized successfully. Ready to make API calls.")

except Exception as e:

    print(f"Error initializing Nebius AI client: {e}")
    print("!!! Execution cannot proceed without a valid client. Please check your API key and network connection. !!!")

    client = None

print("Client setup step complete.")
print("-" * 25)
```

```output
Attempting to initialize the Nebius AI client...
Nebius AI client initialized successfully. Ready to make API calls.
Client setup step complete.
-------------------------
```

### 7. Core Component: Cosine Similarity Function

To evaluate how semantically similar the generated answer is to our ground truth answer, we use **Cosine Similarity**. This metric measures the cosine of the angle between two vectors (in our case, the embedding vectors of the two answers).

- A score of **1** means the vectors point in the same direction (maximum similarity).
- A score of **0** means the vectors are orthogonal (no similarity).
- A score of **-1** means the vectors point in opposite directions (maximum dissimilarity).

For text embeddings, scores typically range from 0 to 1, where higher values indicate greater semantic similarity.

We define a function `calculate_cosine_similarity` that takes two text strings, generates their embeddings using the Nebius client, and returns their cosine similarity score.

```python
def calculate_cosine_similarity(text1, text2, client, embedding_model):
    """Calculates cosine similarity between the embeddings of two texts.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        client (OpenAI): The initialized Nebius AI client.
        embedding_model (str): The name of the embedding model to use.

    Returns:
        float: The cosine similarity score (between 0.0 and 1.0), or 0.0 if an error occurs.
    """
    if not client:
        print("  Error: Nebius client not available for similarity calculation.")
        return 0.0
    if not text1 or not text2:

        return 0.0

    try:

        response = client.embeddings.create(model=embedding_model, input=[text1, text2])


        embedding1 = np.array(response.data[0].embedding)
        embedding2 = np.array(response.data[1].embedding)


        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)



        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]


        return max(0.0, min(1.0, similarity_score))

    except Exception as e:
        print(f"  Error calculating cosine similarity: {e}")
        return 0.0


print("Defining the 'calculate_cosine_similarity' function.")
if client:
    test_sim = calculate_cosine_similarity("apple", "orange", client, NEBIUS_EMBEDDING_MODEL)
    print(f"Testing similarity function: Similarity between 'apple' and 'orange' = {test_sim:.2f}")
else:
    print("Skipping similarity function test as Nebius client is not initialized.")
print("-" * 25)
```

```output
Defining the 'calculate_cosine_similarity' function.
Testing similarity function: Similarity between 'apple' and 'orange' = 0.77
-------------------------
```

### 8. The Experiment: Iterating Through Configurations

This section contains the main experimental loop. We will systematically iterate through all combinations of the tuning parameters we defined earlier (`CHUNK_SIZES_TO_TEST`, `CHUNK_OVERLAPS_TO_TEST`, `RETRIEVAL_TOP_K_TO_TEST`).

**Workflow for Each Parameter Combination:**

1.  **Prepare Data (Chunking/Embedding/Indexing - Step 8.1):**
    *   **Check if Re-computation Needed:** If the `chunk_size` or `chunk_overlap` has changed from the previous iteration, we need to re-process the corpus.
    *   **Chunking:** Split all documents in `corpus_texts` using the current `chunk_size` and `chunk_overlap` via the `chunk_text` function.
    *   **Embedding:** Convert each text chunk into a numerical vector (embedding) using the specified Nebius embedding model (`NEBIUS_EMBEDDING_MODEL`). We do this in batches for efficiency.
    *   **Indexing:** Build a FAISS index (`IndexFlatL2`) from the generated embeddings. FAISS allows for very fast searching to find the chunks whose embeddings are most similar to the query embedding.
    *   *Optimization:* If chunk settings haven't changed, we reuse the existing chunks, embeddings, and index from the previous iteration to save time and API calls.

2.  **Test RAG Strategies (Step 8.2):**
    *   For the current `top_k` value, run each of the defined RAG strategies:
        *   **Simple RAG:** Retrieve `top_k` chunks based on similarity to the original query.
        *   **Query Rewrite RAG:** First, ask the LLM to rewrite the original query to be potentially better for vector search. Then, retrieve `top_k` chunks based on similarity to the *rewritten* query.
        *   **Rerank RAG (Simulated):** Retrieve more chunks initially (`top_k * RERANK_RETRIEVAL_MULTIPLIER`). Then, *simulate* reranking by simply taking the top `top_k` results from this larger initial set. (A real implementation would use a more sophisticated reranking model).

3.  **Evaluate & Store Results (Step 8.3 within `run_and_evaluate`):**
    *   For each strategy run:
        *   **Retrieve:** Find the relevant chunk indices using the FAISS index.
        *   **Generate:** Construct a prompt containing the retrieved chunk(s) as context and the *original* `test_query`. Send this to the Nebius generation model (`NEBIUS_GENERATION_MODEL`) to get the final answer.
        *   **Evaluate (Faithfulness):** Use the LLM evaluator (`NEBIUS_EVALUATION_MODEL`) with the `FAITHFULNESS_PROMPT` to score how well the generated answer aligns with the `true_answer_for_query`.
        *   **Evaluate (Relevancy):** Use the LLM evaluator with the `RELEVANCY_PROMPT` to score how well the generated answer addresses the `test_query`.
        *   **Evaluate (Similarity):** Use our `calculate_cosine_similarity` function to get the semantic similarity score between the generated answer and the `true_answer_for_query`.
        *   **Calculate Average Score:** Compute the average of Faithfulness, Relevancy, and Similarity scores.
        *   **Record:** Store all parameters (`chunk_size`, `overlap`, `top_k`, `strategy`), the retrieved indices, the rewritten query (if applicable), the generated answer, the individual scores, the average score, and the execution time for this specific run.

We use `tqdm` to display a progress bar for the outer loop iterating through parameter combinations.

```python

all_results = []



last_chunk_size = -1
last_overlap = -1
current_index = None
current_chunks = []
current_embeddings = None


if not client:
    print("STOPPING: Nebius AI client is not initialized. Cannot run experiment.")
else:
    print("=== Starting RAG Experiment Loop ===\n")


    param_combinations = list(itertools.product(
        CHUNK_SIZES_TO_TEST,
        CHUNK_OVERLAPS_TO_TEST,
        RETRIEVAL_TOP_K_TO_TEST
    ))

    print(f"Total parameter combinations to test: {len(param_combinations)}")




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
                         print(f"  Warning: No chunks created for document {doc_index} with size={chunk_size}, overlap={chunk_overlap}. Skipping document.")
                         continue
                    temp_chunks.extend(doc_chunks)

                current_chunks = temp_chunks
                if not current_chunks:

                    raise ValueError("No chunks were created for the current configuration.")

            except Exception as e:
                 print(f"    ERROR during chunking for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this configuration.")
                 last_chunk_size, last_overlap = -1, -1
                 continue




            try:
                batch_size = 32
                temp_embeddings = []


                for i in range(0, len(current_chunks), batch_size):
                    batch_texts = current_chunks[i : min(i + batch_size, len(current_chunks))]

                    response = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=batch_texts)

                    batch_embeddings = [item.embedding for item in response.data]
                    temp_embeddings.extend(batch_embeddings)
                    time.sleep(0.05)


                current_embeddings = np.array(temp_embeddings)

                if current_embeddings.ndim != 2 or current_embeddings.shape[0] != len(current_chunks):
                    raise ValueError(f"Embeddings array shape mismatch. Expected ({len(current_chunks)}, dim), Got {current_embeddings.shape}")


            except Exception as e:
                print(f"    ERROR generating embeddings for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this chunk config.")

                last_chunk_size, last_overlap = -1, -1
                current_chunks = []
                current_embeddings = None
                continue




            try:
                embedding_dim = current_embeddings.shape[1]




                current_index = faiss.IndexFlatL2(embedding_dim)

                current_index.add(current_embeddings.astype('float32'))

                if current_index.ntotal == 0:
                     raise ValueError("FAISS index is empty after adding vectors. No vectors were added.")

            except Exception as e:
                print(f"    ERROR building FAISS index for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this chunk config.")

                last_chunk_size, last_overlap = -1, -1
                current_index = None
                current_embeddings = None
                current_chunks = []
                continue





        if current_index is None or not current_chunks:
            print(f"    WARNING: Index or chunks not available for Size={chunk_size}, Overlap={chunk_overlap}. Skipping Top-K={top_k} test.")
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



                query_embedding_response = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=[query_to_use])
                query_embedding = query_embedding_response.data[0].embedding
                query_vector = np.array([query_embedding]).astype('float32')



                actual_k = min(k_for_search, current_index.ntotal)
                if actual_k == 0:
                    raise ValueError("Index is empty or k_for_search is zero, cannot search.")


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
                    print(f"      Warning: No relevant chunks found for {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}). Setting answer to indicate this.")
                    result['answer'] = "No relevant context found in the documents based on the query."

                else:


                    context_str = "\n\n".join(retrieved_chunks)


                    sys_prompt_gen = "You are a helpful AI assistant. Answer the user's query based strictly on the provided context. If the context doesn't contain the answer, state that clearly. Be concise."



                    user_prompt_gen = f"Context:\n------\n{context_str}\n------\n\nQuery: {test_query}\n\nAnswer:"


                    gen_response = client.chat.completions.create(
                        model=NEBIUS_GENERATION_MODEL,
                        messages=[
                            {"role": "system", "content": sys_prompt_gen},
                            {"role": "user", "content": user_prompt_gen}
                        ],
                        temperature=GENERATION_TEMPERATURE,
                        max_tokens=GENERATION_MAX_TOKENS,
                        top_p=GENERATION_TOP_P
                    )

                    generated_answer = gen_response.choices[0].message.content.strip()
                    result['answer'] = generated_answer








                    eval_params = {'model': NEBIUS_EVALUATION_MODEL, 'temperature': 0.0, 'max_tokens': 10}


                    prompt_f = FAITHFULNESS_PROMPT.format(question=test_query, response=generated_answer, true_answer=true_answer_for_query)
                    try:
                        resp_f = client.chat.completions.create(messages=[{"role": "user", "content": prompt_f}], **eval_params)

                        result['faithfulness'] = max(0.0, min(1.0, float(resp_f.choices[0].message.content.strip())))
                    except Exception as eval_e:
                        print(f"      Warning: Faithfulness score parsing error for {strategy_name} - {eval_e}. Score set to 0.0")
                        result['faithfulness'] = 0.0


                    prompt_r = RELEVANCY_PROMPT.format(question=test_query, response=generated_answer)
                    try:
                        resp_r = client.chat.completions.create(messages=[{"role": "user", "content": prompt_r}], **eval_params)

                        result['relevancy'] = max(0.0, min(1.0, float(resp_r.choices[0].message.content.strip())))
                    except Exception as eval_e:
                        print(f"      Warning: Relevancy score parsing error for {strategy_name} - {eval_e}. Score set to 0.0")
                        result['relevancy'] = 0.0


                    result['similarity_score'] = calculate_cosine_similarity(
                        generated_answer,
                        true_answer_for_query,
                        client,
                        NEBIUS_EMBEDDING_MODEL
                    )


                    result['avg_score'] = (result['faithfulness'] + result['relevancy'] + result['similarity_score']) / 3.0

            except Exception as e:

                error_message = f"ERROR during {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}): {str(e)[:200]}..."
                print(f"    {error_message}")
                result['answer'] = error_message

                result['faithfulness'] = 0.0
                result['relevancy'] = 0.0
                result['similarity_score'] = 0.0
                result['avg_score'] = 0.0


            run_end_time = time.time()
            result['time_sec'] = run_end_time - run_start_time


            print(f"    Finished: {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}). AvgScore={result['avg_score']:.2f}, Time={result['time_sec']:.2f}s")
            return result





        result_simple = run_and_evaluate("Simple RAG", test_query, top_k)
        all_results.append(result_simple)


        rewritten_q = test_query
        try:


             sys_prompt_rw = "You are an expert query optimizer. Rewrite the user's query to be ideal for vector database retrieval. Focus on key entities, concepts, and relationships. Remove conversational fluff. Output ONLY the rewritten query text."
             user_prompt_rw = f"Original Query: {test_query}\n\nRewritten Query:"


             resp_rw = client.chat.completions.create(
                 model=NEBIUS_GENERATION_MODEL,
                 messages=[
                     {"role": "system", "content": sys_prompt_rw},
                     {"role": "user", "content": user_prompt_rw}
                 ],
                 temperature=0.1,
                 max_tokens=100,
                 top_p=0.9
             )

             candidate_q = resp_rw.choices[0].message.content.strip()

             candidate_q = re.sub(r'^(rewritten query:|query:)\s*', '', candidate_q, flags=re.IGNORECASE).strip('"')


             if candidate_q and len(candidate_q) > 5 and candidate_q.lower() != test_query.lower():
                 rewritten_q = candidate_q



        except Exception as e:
             print(f"    Warning: Error during query rewrite: {e}. Using original query.")
             rewritten_q = test_query


        result_rewrite = run_and_evaluate("Query Rewrite RAG", rewritten_q, top_k)
        all_results.append(result_rewrite)



        result_rerank = run_and_evaluate("Rerank RAG (Simulated)", test_query, top_k, use_simulated_rerank=True)
        all_results.append(result_rerank)

    print("\n=== RAG Experiment Loop Finished ===")
    print("-" * 25)
```

```output
=== Starting RAG Experiment Loop ===

Total parameter combinations to test: 8
```

```output
Testing Configurations:   0%|          | 0/8 [00:00<?, ?it/s]
```

```output
    Finished: Simple RAG (C=150, O=30, K=3). AvgScore=0.89, Time=609.06s
    Finished: Query Rewrite RAG (C=150, O=30, K=3). AvgScore=0.89, Time=10.36s
    Finished: Rerank RAG (Simulated) (C=150, O=30, K=3). AvgScore=0.89, Time=9.53s
    Finished: Simple RAG (C=150, O=30, K=5). AvgScore=0.89, Time=8.40s
    Finished: Query Rewrite RAG (C=150, O=30, K=5). AvgScore=0.89, Time=8.36s
    Finished: Rerank RAG (Simulated) (C=150, O=30, K=5). AvgScore=0.89, Time=8.34s
    Finished: Simple RAG (C=150, O=50, K=3). AvgScore=0.89, Time=9.78s
    Finished: Query Rewrite RAG (C=150, O=50, K=3). AvgScore=0.89, Time=9.68s
    Finished: Rerank RAG (Simulated) (C=150, O=50, K=3). AvgScore=0.89, Time=8.43s
    Finished: Simple RAG (C=150, O=50, K=5). AvgScore=0.89, Time=9.74s
    Finished: Query Rewrite RAG (C=150, O=50, K=5). AvgScore=0.89, Time=9.39s
    Finished: Rerank RAG (Simulated) (C=150, O=50, K=5). AvgScore=0.89, Time=8.53s
    Finished: Simple RAG (C=250, O=30, K=3). AvgScore=0.89, Time=9.36s
    Finished: Query Rewrite RAG (C=250, O=30, K=3). AvgScore=0.89, Time=8.36s
    Finished: Rerank RAG (Simulated) (C=250, O=30, K=3). AvgScore=0.89, Time=9.56s
    Finished: Simple RAG (C=250, O=30, K=5). AvgScore=0.89, Time=8.77s
    Finished: Query Rewrite RAG (C=250, O=30, K=5). AvgScore=0.89, Time=9.63s
    Finished: Rerank RAG (Simulated) (C=250, O=30, K=5). AvgScore=0.89, Time=6.63s
    Finished: Simple RAG (C=250, O=50, K=3). AvgScore=0.90, Time=8.98s
    Finished: Query Rewrite RAG (C=250, O=50, K=3). AvgScore=0.90, Time=6.55s
    Finished: Rerank RAG (Simulated) (C=250, O=50, K=3). AvgScore=0.89, Time=41.23s
    Finished: Simple RAG (C=250, O=50, K=5). AvgScore=0.89, Time=6.93s
    Finished: Query Rewrite RAG (C=250, O=50, K=5). AvgScore=0.89, Time=6.11s
    Finished: Rerank RAG (Simulated) (C=250, O=50, K=5). AvgScore=0.89, Time=7.09s

=== RAG Experiment Loop Finished ===
-------------------------
```

### 9. Analysis: Reviewing the Results

Now that the experiment loop has completed and `all_results` contains the data from each run, we'll use the Pandas library to analyze the findings.

1.  **Create DataFrame:** Convert the list of result dictionaries (`all_results`) into a Pandas DataFrame for easy manipulation and viewing.
2.  **Sort Results:** Sort the DataFrame by the `avg_score` (the average of Faithfulness, Relevancy, and Similarity) in descending order, so the best-performing configurations appear first.
3.  **Display Top Configurations:** Show the top N rows of the sorted DataFrame, including key parameters, scores, and the generated answer, to quickly identify promising settings.
4.  **Summarize Best Run:** Print a clear summary of the single best-performing configuration based on the average score, showing its parameters, individual scores, time taken, and the full answer it generated.

```python
print("--- Analyzing Experiment Results ---")


if not all_results:
    print("No results were generated during the experiment. Cannot perform analysis.")
else:

    results_df = pd.DataFrame(all_results)
    print(f"Total results collected: {len(results_df)}")



    results_df_sorted = results_df.sort_values(by='avg_score', ascending=False).reset_index(drop=True)

    print("\n--- Top 10 Performing Configurations (Sorted by Average Score) ---")

    display_cols = [
        'chunk_size', 'overlap', 'top_k', 'strategy',
        'avg_score', 'faithfulness', 'relevancy', 'similarity_score',
        'time_sec',
        'answer'
    ]

    display_cols = [col for col in display_cols if col in results_df_sorted.columns]



    display(results_df_sorted[display_cols].head(10))


    print("\n--- Best Configuration Summary ---")

    if not results_df_sorted.empty:

        best_run = results_df_sorted.iloc[0]


        print(f"Chunk Size: {best_run.get('chunk_size', 'N/A')} words")
        print(f"Overlap: {best_run.get('overlap', 'N/A')} words")
        print(f"Top-K Retrieved: {best_run.get('top_k', 'N/A')} chunks")
        print(f"Strategy: {best_run.get('strategy', 'N/A')}")

        avg_score = best_run.get('avg_score', 0.0)
        faithfulness = best_run.get('faithfulness', 0.0)
        relevancy = best_run.get('relevancy', 0.0)
        similarity = best_run.get('similarity_score', 0.0)
        time_sec = best_run.get('time_sec', 0.0)
        best_answer = best_run.get('answer', 'N/A')

        print(f"---> Average Score (Faith+Rel+Sim): {avg_score:.3f}")
        print(f"      (Faithfulness: {faithfulness:.3f}, Relevancy: {relevancy:.3f}, Similarity: {similarity:.3f})")
        print(f"Time Taken: {time_sec:.2f} seconds")
        print(f"\nBest Answer Generated:")

        print(best_answer)
    else:

        print("Could not determine the best configuration (no valid results found).")

print("\n--- Analysis Complete --- ")
```

```output
--- Analyzing Experiment Results ---
Total results collected: 24

--- Top 10 Performing Configurations (Sorted by Average Score) ---
```

```output
   chunk_size  overlap  top_k                strategy  avg_score  \
0         250       50      3              Simple RAG   0.899417
1         250       50      3       Query Rewrite RAG   0.896859
2         150       30      3  Rerank RAG (Simulated)   0.894125
3         150       50      3       Query Rewrite RAG   0.893823
4         150       30      3       Query Rewrite RAG   0.893666
5         150       50      3              Simple RAG   0.892774
6         250       50      3  Rerank RAG (Simulated)   0.891570
7         250       30      3       Query Rewrite RAG   0.890878
8         250       30      5              Simple RAG   0.890867
9         150       50      5              Simple RAG   0.890656

   faithfulness  relevancy  similarity_score   time_sec  \
0           0.9        1.0          0.798251   8.975824
1           0.9        1.0          0.790578   6.550637
2           0.9        1.0          0.782374   9.526656
3           0.9        1.0          0.781468   9.675948
4           0.9        1.0          0.780997  10.357061
5           0.9        1.0          0.778321   9.777294
6           0.9        1.0          0.774709  41.228211
7           0.9        1.0          0.772635   8.359087
8           0.9        1.0          0.772601   8.767287
9           0.9        1.0          0.771967   9.743746

                                                                                                                                                  answer
0  Solar power and hydropower differ significantly in consistency and environmental impact:\n\n- **Consistency**:  \n  - **Solar Power**: Inconsisten...
1  **Consistency:**\n- **Hydropower** is highly reliable and provides consistent, large-scale power, as it is not dependent on weather conditions onc...
2  **Consistency:**  \n- **Hydropower** is highly reliable and consistent, providing large-scale power 24/7, as it is not dependent on weather or tim...
3  **Consistency:**  \n- **Hydropower** is highly reliable and provides consistent, large-scale power, as it is not dependent on weather conditions o...
4  **Consistency:**\n- **Hydropower** is highly reliable and consistent, providing large-scale power 24/7, as it relies on the continuous flow of wat...
5  **Consistency:**  \n- **Hydropower** is highly consistent and reliable, providing large-scale power 24/7, especially with dams that store water fo...
6  **Consistency:**  \n- **Hydropower** is highly reliable and provides consistent, large-scale power, especially with dams that can store water and ...
7  **Consistency:**  \n- **Hydropower** is highly reliable and provides consistent, large-scale power, especially with dams that can store water and ...
8  **Consistency:**  \n- **Solar Power:** Inconsistent due to dependence on weather and daylight. Requires storage solutions (e.g., batteries) for re...
9  **Consistency:**  \n- **Solar Power:** Inconsistent due to dependence on weather and daylight. Requires storage solutions (e.g., batteries) for re...
```

```output

--- Best Configuration Summary ---
Chunk Size: 250 words
Overlap: 50 words
Top-K Retrieved: 3 chunks
Strategy: Simple RAG
---> Average Score (Faith+Rel+Sim): 0.899
      (Faithfulness: 0.900, Relevancy: 1.000, Similarity: 0.798)
Time Taken: 8.98 seconds

Best Answer Generated:
Solar power and hydropower differ significantly in consistency and environmental impact:

- **Consistency**:
  - **Solar Power**: Inconsistent, as it depends on weather conditions and time of day. Requires storage solutions (like batteries) for reliable supply.
  - **Hydropower**: Highly consistent, providing large-scale, reliable power 24/7, especially with dams.

- **Environmental Impact**:
  - **Solar Power**: Clean with minimal emissions during operation, but manufacturing panels and disposal can have environmental impacts.
  - **Hydropower**: Large dams can severely harm ecosystems, disrupt fish migration, and displace communities. Run-of-river systems are less disruptive but still impact local environments.

In summary, hydropower is more consistent but has greater environmental risks, while solar power is cleaner but less reliable without storage.

--- Analysis Complete ---
```

### 10. Conclusion: What Did We Learn?

We have successfully constructed and executed an end-to-end pipeline to experiment with various RAG configurations and evaluate their performance using multiple metrics on the Nebius AI platform.

By examining the results table and the best configuration summary above, we can gain insights specific to *our chosen corpus, query, and models*.

**Reflection Points:**

*   **Chunking Impact:** Did a specific `chunk_size` or `overlap` tend to produce better average scores? Consider why smaller chunks might capture specific facts better, while larger chunks might provide more context. How did overlap seem to influence the results?
*   **Retrieval Quantity (`top_k`):** How did increasing `top_k` affect the scores? Did retrieving more chunks always lead to better answers, or did it sometimes introduce noise or irrelevant information, potentially lowering faithfulness or similarity?
*   **Strategy Comparison:** Did the 'Query Rewrite' or 'Rerank (Simulated)' strategies offer a consistent advantage over 'Simple RAG' in terms of the average score? Was the potential improvement significant enough to justify the extra steps (e.g., additional LLM call for rewrite, larger initial retrieval for rerank)?
*   **Evaluation Metrics:**
    *   Look at the 'Best Answer' and compare it to the `true_answer_for_query`. Do the individual scores (Faithfulness, Relevancy, Similarity) seem to reflect the quality you perceive?
    *   Did high similarity always correlate with high faithfulness? Could an answer be similar but unfaithful, or faithful but dissimilar?
    *   How reliable do you feel the automated LLM evaluation (Faithfulness, Relevancy) is compared to the more objective Cosine Similarity? What are the potential limitations of LLM-based evaluation (e.g., sensitivity to prompt wording, model biases)?
*   **Overall Performance:** Did any configuration achieve a near-perfect average score? What might be preventing a perfect score (e.g., limitations of the source documents, inherent ambiguity in language, imperfect retrieval)?

**Key Takeaway:** Optimizing a RAG system is an iterative process. The best configuration often depends heavily on the specific dataset, the nature of the user queries, the chosen embedding and LLM models, and the evaluation criteria. Systematic experimentation, like the process followed in this notebook, is crucial for finding settings that perform well for a particular use case.

**Potential Next Steps & Further Exploration:**

*   **Expand Test Parameters:** Try a wider range of `chunk_size`, `overlap`, and `top_k` values.
*   **Different Queries:** Test the same configurations with different types of queries (e.g., fact-based, comparison, summarization) to see how performance varies.
*   **Larger/Different Corpus:** Use a more extensive or domain-specific knowledge base.
*   **Implement True Reranking:** Replace the simulated reranking with a dedicated cross-encoder reranking model (e.g., from Hugging Face Transformers or Cohere Rerank) to re-score the initially retrieved documents based on relevance.
*   **Alternative Models:** Experiment with different Nebius AI models for embedding, generation, or evaluation to see their impact.
*   **Advanced Chunking:** Explore more sophisticated chunking strategies (e.g., recursive character splitting, semantic chunking).
*   **Human Evaluation:** Complement the automated metrics with human judgment for a more nuanced understanding of answer quality.
