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
# Install libraries (run this cell only once if needed)
# !pip install openai pandas numpy faiss-cpu ipywidgets tqdm scikit-learn
```

**Remember!** After the installation finishes, you might need to **Restart the Kernel** (or Runtime) for Jupyter/Colab to recognize the newly installed packages. Look for this option in the menu (e.g., 'Kernel' -> 'Restart Kernel...' or 'Runtime' -> 'Restart Runtime').

### 2. Setup: Importing Libraries

With the libraries installed, we import them into our Python environment to make their functions available.

```python
import os                     # For accessing environment variables (like API keys)
import time                   # For timing operations
import re                     # For regular expressions (text cleaning)
import warnings               # For controlling warning messages
import itertools              # For creating parameter combinations easily
import getpass                # For securely prompting for API keys if not set

import numpy as np            # Numerical library for vector operations
import pandas as pd           # Data manipulation library for tables (DataFrames)
import faiss                  # Library for fast vector similarity search
from openai import OpenAI     # Client library for Nebius API interaction
from tqdm.notebook import tqdm # Library for displaying progress bars
from sklearn.metrics.pairwise import cosine_similarity # For calculating similarity score

# Configure display options for Pandas DataFrames for better readability
pd.set_option('display.max_colwidth', 150) # Show more text content in table cells
pd.set_option('display.max_rows', 100)     # Display more rows in tables
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress specific non-critical warnings

print("Libraries imported successfully!")
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
# --- NebiusAI API Configuration ---
# It's best practice to store API keys as environment variables rather than hardcoding them.
# Provide your actual key here or set it as an environment variable
NEBIUS_API_KEY = os.getenv('NEBIUS_API_KEY', None)  # Load API key from environment variable
if NEBIUS_API_KEY is None:
    print("Warning: NEBIUS_API_KEY not set. Please set it in your environment variables or provide it directly in the code.") 
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/" 
NEBIUS_EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"  # Model for converting text to vector embeddings
NEBIUS_GENERATION_MODEL = "deepseek-ai/DeepSeek-V3"    # LLM for generating the final answers
NEBIUS_EVALUATION_MODEL = "deepseek-ai/DeepSeek-V3"    # LLM used for evaluating the generated answers

# --- Text Generation Parameters (for RAG answer generation) ---
GENERATION_TEMPERATURE = 0.1  # Lower values (e.g., 0.1-0.3) make output more focused and deterministic, good for fact-based answers.
GENERATION_MAX_TOKENS = 400   # Maximum number of tokens (roughly words/sub-words) in the generated answer.
GENERATION_TOP_P = 0.9        # Nucleus sampling parameter (alternative to temperature, usually fine at default).

# --- Evaluation Prompts (Instructions for the Evaluator LLM) ---
# Faithfulness: Does the answer stay true to the provided context?
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

# Relevancy: Does the answer directly address the user's query?
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

# --- Parameters to Tune (The experimental variables) ---
CHUNK_SIZES_TO_TEST = [150, 250]    # List of chunk sizes (in words) to experiment with.
CHUNK_OVERLAPS_TO_TEST = [30, 50]   # List of chunk overlaps (in words) to experiment with.
RETRIEVAL_TOP_K_TO_TEST = [3, 5]   # List of 'k' values (number of chunks to retrieve) to test.

# --- Reranking Configuration (Only used for the Rerank strategy) ---
RERANK_RETRIEVAL_MULTIPLIER = 3 # For simulated reranking: retrieve K * multiplier chunks initially.

# --- Validate API Key --- 
print("--- Configuration Check --- ")
print(f"Attempting to load Nebius API Key from environment variable 'NEBIUS_API_KEY'...")
if not NEBIUS_API_KEY:
    print("Nebius API Key not found in environment variables.")
    # Prompt the user securely if the key is not found.
    NEBIUS_API_KEY = getpass.getpass("Please enter your Nebius API Key: ")
else:
    print("Nebius API Key loaded successfully from environment variable.")

# Print a summary of key settings for verification
print(f"Models: Embed='{NEBIUS_EMBEDDING_MODEL}', Gen='{NEBIUS_GENERATION_MODEL}', Eval='{NEBIUS_EVALUATION_MODEL}'")
print(f"Chunk Sizes to Test: {CHUNK_SIZES_TO_TEST}")
print(f"Overlaps to Test: {CHUNK_OVERLAPS_TO_TEST}")
print(f"Top-K Values to Test: {RETRIEVAL_TOP_K_TO_TEST}")
print(f"Generation Temp: {GENERATION_TEMPERATURE}, Max Tokens: {GENERATION_MAX_TOKENS}")
print("Configuration ready.")
print("-" * 25)
```

### 4. Input Data: The Knowledge Source & Our Question

Every RAG system needs a knowledge base to draw information from. Here, we define:
*   `corpus_texts`: A list of strings, where each string is a document containing information (in this case, about renewable energy sources).
*   `test_query`: The specific question we want the RAG system to answer using the `corpus_texts`.
*   `true_answer_for_query`: A carefully crafted 'ground truth' answer based *only* on the information available in `corpus_texts`. This is essential for evaluating Faithfulness and Semantic Similarity accurately.

```python
# Our knowledge base: A list of text documents about renewable energy
corpus_texts = [
    "Solar power uses PV panels or CSP systems. PV converts sunlight directly to electricity. CSP uses mirrors to heat fluid driving a turbine. It's clean but varies with weather/time. Storage (batteries) is key for consistency.", # Doc 0
    "Wind energy uses turbines in wind farms. It's sustainable with low operating costs. Wind speed varies, siting can be challenging (visual/noise). Offshore wind is stronger and more consistent.", # Doc 1
    "Hydropower uses moving water, often via dams spinning turbines. Reliable, large-scale power with flood control/water storage benefits. Big dams harm ecosystems and displace communities. Run-of-river is smaller, less disruptive.", # Doc 2
    "Geothermal energy uses Earth's heat via steam/hot water for turbines. Consistent 24/7 power, small footprint. High initial drilling costs, sites are geographically limited.", # Doc 3
    "Biomass energy from organic matter (wood, crops, waste). Burned directly or converted to biofuels. Uses waste, provides dispatchable power. Requires sustainable sourcing. Combustion releases emissions (carbon-neutral if balanced by regrowth)." # Doc 4
]

# The question we will ask the RAG system
test_query = "Compare the consistency and environmental impact of solar power versus hydropower."

# !!! CRITICAL: The 'True Answer' MUST be derivable ONLY from the corpus_texts above !!!
# This is our ground truth for evaluation.
true_answer_for_query = "Solar power's consistency varies with weather and time of day, requiring storage like batteries. Hydropower is generally reliable, but large dams have significant environmental impacts on ecosystems and communities, unlike solar power's primary impact being land use for panels."

print(f"Loaded {len(corpus_texts)} documents into our corpus.")
print(f"Test Query: '{test_query}'")
print(f"Reference (True) Answer for evaluation: '{true_answer_for_query}'")
print("Input data is ready.")
print("-" * 25)
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
    words = text.split()      # Split the text into a list of individual words
    total_words = len(words) # Calculate the total number of words in the text
    chunks = []             # Initialize an empty list to store the generated chunks
    start_index = 0         # Initialize the starting word index for the first chunk

    # --- Input Validation ---
    # Ensure chunk_size is a positive integer.
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        print(f"  Warning: Invalid chunk_size ({chunk_size}). Must be a positive integer. Returning the whole text as one chunk.")
        return [text]
    # Ensure chunk_overlap is a non-negative integer smaller than chunk_size.
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        print(f"  Warning: Invalid chunk_overlap ({chunk_overlap}). Must be a non-negative integer. Setting overlap to 0.")
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        # If overlap is too large, adjust it to a reasonable fraction (e.g., 1/3) of chunk_size
        # This prevents infinite loops or nonsensical chunking.
        adjusted_overlap = chunk_size // 3
        print(f"  Warning: chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Adjusting overlap to {adjusted_overlap}.")
        chunk_overlap = adjusted_overlap

    # --- Chunking Loop ---
    # Continue chunking as long as the start_index is within the bounds of the text
    while start_index < total_words:
        # Determine the end index for the current chunk.
        # It's the minimum of (start + chunk_size) and the total number of words.
        end_index = min(start_index + chunk_size, total_words)
        
        # Extract the words for the current chunk and join them back into a single string.
        current_chunk_text = " ".join(words[start_index:end_index])
        chunks.append(current_chunk_text) # Add the generated chunk to the list
        
        # Calculate the starting index for the *next* chunk.
        # Move forward by (chunk_size - chunk_overlap) words.
        next_start_index = start_index + chunk_size - chunk_overlap
        
        # --- Safety Checks ---
        # Check 1: Prevent infinite loops if overlap causes no progress.
        # This can happen if chunk_size is very small or overlap is very large relative to chunk_size.
        if next_start_index <= start_index:
            if end_index == total_words: # If we are already at the end, we can safely break.
                break
            else: 
                # Force progress by moving forward by at least one word.
                print(f"  Warning: Chunking logic stuck (start={start_index}, next_start={next_start_index}). Forcing progress.")
                next_start_index = start_index + 1 
                
        # Check 2: If the calculated next start index is already at or beyond the total number of words, we are done.
        if next_start_index >= total_words:
            break
            
        # Move the start_index to the calculated position for the next iteration.
        start_index = next_start_index
        
    return chunks # Return the complete list of text chunks

# --- Quick Test ---
# Test the function with the first document and sample parameters.
print("Defining the 'chunk_text' function.")
sample_chunk_size = 150
sample_overlap = 30
sample_chunks = chunk_text(corpus_texts[0], sample_chunk_size, sample_overlap) 
print(f"Test chunking on first doc (size={sample_chunk_size} words, overlap={sample_overlap} words): Created {len(sample_chunks)} chunks.")
if sample_chunks: # Only print if chunks were created
    print(f"First sample chunk:\n'{sample_chunks[0]}'")
print("-" * 25)
```

### 6. Core Component: Connecting to Nebius AI

To use the Nebius AI models (for embedding, generation, evaluation), we need to establish a connection to their API. We use the `openai` Python library, which provides a convenient way to interact with OpenAI-compatible APIs like Nebius.

We instantiate an `OpenAI` client object, providing our API key and the specific Nebius API endpoint URL.

```python
client = None # Initialize client variable to None globally

print("Attempting to initialize the Nebius AI client...")
try:
    # Check if the API key is actually available before creating the client
    if not NEBIUS_API_KEY:
        raise ValueError("Nebius API Key is missing. Cannot initialize client.")
        
    # Create the OpenAI client object, configured for the Nebius API.
    client = OpenAI(
        api_key=NEBIUS_API_KEY,     # Pass the API key loaded earlier
        base_url=NEBIUS_BASE_URL  # Specify the Nebius API endpoint
    )
    
    # Optional: Add a quick test call to verify the client connection,
    # e.g., listing models (if supported and desired). This might incur costs.
    # try:
    #     client.models.list() 
    #     print("Client connection verified by listing models.")
    # except Exception as test_e:
    #     print(f"Warning: Could not verify client connection with test call: {test_e}")
    
    print("Nebius AI client initialized successfully. Ready to make API calls.")
    
except Exception as e:
    # Catch any errors during client initialization (e.g., invalid key, network issues)
    print(f"Error initializing Nebius AI client: {e}")
    print("!!! Execution cannot proceed without a valid client. Please check your API key and network connection. !!!")
    # Setting client back to None to prevent further attempts if initialization failed
    client = None 

print("Client setup step complete.")
print("-" * 25)
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
        # Handle cases where one or both texts might be empty or None
        return 0.0
        
    try:
        # Generate embeddings for both texts in a single API call if possible
        response = client.embeddings.create(model=embedding_model, input=[text1, text2])
        
        # Extract the embedding vectors
        embedding1 = np.array(response.data[0].embedding)
        embedding2 = np.array(response.data[1].embedding)
        
        # Reshape vectors to be 2D arrays as expected by cosine_similarity
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity using scikit-learn
        # cosine_similarity returns a 2D array, e.g., [[similarity]], so we extract the value.
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Clamp the score between 0.0 and 1.0 for safety/consistency
        return max(0.0, min(1.0, similarity_score))
        
    except Exception as e:
        print(f"  Error calculating cosine similarity: {e}")
        return 0.0 # Return 0.0 in case of any API or calculation errors

# --- Quick Test ---
print("Defining the 'calculate_cosine_similarity' function.")
if client: # Only run test if client is initialized
    test_sim = calculate_cosine_similarity("apple", "orange", client, NEBIUS_EMBEDDING_MODEL)
    print(f"Testing similarity function: Similarity between 'apple' and 'orange' = {test_sim:.2f}")
else:
    print("Skipping similarity function test as Nebius client is not initialized.")
print("-" * 25)
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
# List to store the detailed results from each experimental run
all_results = []

# --- Cache variables for Chunking/Embedding/Indexing --- 
# These variables help us avoid redundant computations when only 'top_k' changes.
last_chunk_size = -1      # Stores the chunk_size used in the previous iteration
last_overlap = -1         # Stores the chunk_overlap used in the previous iteration
current_index = None      # Holds the active FAISS index
current_chunks = []       # Holds the list of text chunks for the active settings
current_embeddings = None # Holds the numpy array of embeddings for the active chunks

# Check if the Nebius client was initialized successfully before starting
if not client:
    print("STOPPING: Nebius AI client is not initialized. Cannot run experiment.")
else:
    print("=== Starting RAG Experiment Loop ===\n")
    
    # Create all possible combinations of the tuning parameters
    param_combinations = list(itertools.product(
        CHUNK_SIZES_TO_TEST,
        CHUNK_OVERLAPS_TO_TEST,
        RETRIEVAL_TOP_K_TO_TEST
    ))
    
    print(f"Total parameter combinations to test: {len(param_combinations)}")
    
    # --- Main Loop --- 
    # Iterate through each combination (chunk_size, chunk_overlap, top_k)
    # Use tqdm to display a progress bar.
    for chunk_size, chunk_overlap, top_k in tqdm(param_combinations, desc="Testing Configurations"):
        
        # --- 8.1 Processing a Chunking Configuration --- 
        # Check if chunk settings have changed, requiring re-processing.
        if chunk_size != last_chunk_size or chunk_overlap != last_overlap:
            # Uncomment the line below for more verbose logging during execution
            # print(f"\n--- Processing New Chunk Config: Size={chunk_size}, Overlap={chunk_overlap} ---")
            
            # Update cache variables
            last_chunk_size, last_overlap = chunk_size, chunk_overlap
            # Reset index, chunks, and embeddings for the new configuration
            current_index = None 
            current_chunks = []
            current_embeddings = None
            
            # --- 8.1a: Chunking --- 
            # Apply the chunk_text function to each document in the corpus
            try:
                # print("  Chunking documents...") # Uncomment for verbose logging
                temp_chunks = []
                for doc_index, doc in enumerate(corpus_texts):
                    doc_chunks = chunk_text(doc, chunk_size, chunk_overlap)
                    if not doc_chunks:
                         print(f"  Warning: No chunks created for document {doc_index} with size={chunk_size}, overlap={chunk_overlap}. Skipping document.")
                         continue
                    temp_chunks.extend(doc_chunks)
                
                current_chunks = temp_chunks
                if not current_chunks:
                    # If no chunks were created at all (e.g., due to invalid settings or empty corpus)
                    raise ValueError("No chunks were created for the current configuration.")
                # print(f"    Created {len(current_chunks)} chunks total.") # Uncomment for verbose logging
            except Exception as e:
                 print(f"    ERROR during chunking for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this configuration.")
                 last_chunk_size, last_overlap = -1, -1 # Reset cache state
                 continue # Move to the next parameter combination
            
            # --- 8.1b: Embedding --- 
            # Generate embeddings for all chunks using the Nebius API.
            # print("  Generating embeddings...") # Uncomment for verbose logging
            try:
                batch_size = 32 # Process chunks in batches to avoid overwhelming the API or hitting limits.
                temp_embeddings = [] # Temporary list to store embedding vectors
                
                # Loop through chunks in batches
                for i in range(0, len(current_chunks), batch_size):
                    batch_texts = current_chunks[i : min(i + batch_size, len(current_chunks))]
                    # Make the API call to Nebius for the current batch
                    response = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=batch_texts)
                    # Extract the embedding vectors from the API response
                    batch_embeddings = [item.embedding for item in response.data]
                    temp_embeddings.extend(batch_embeddings)
                    time.sleep(0.05) # Add a small delay between batches to be polite to the API endpoint.
                
                # Convert the list of embeddings into a single NumPy array
                current_embeddings = np.array(temp_embeddings)
                # Basic validation check on the embeddings array
                if current_embeddings.ndim != 2 or current_embeddings.shape[0] != len(current_chunks):
                    raise ValueError(f"Embeddings array shape mismatch. Expected ({len(current_chunks)}, dim), Got {current_embeddings.shape}")
                # print(f"    Generated {current_embeddings.shape[0]} embeddings (Dimension: {current_embeddings.shape[1]}).") # Uncomment for verbose logging

            except Exception as e:
                print(f"    ERROR generating embeddings for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this chunk config.")
                # Reset cache variables to indicate failure for this chunk setting
                last_chunk_size, last_overlap = -1, -1 
                current_chunks = []
                current_embeddings = None
                continue # Skip to the next parameter combination
                
            # --- 8.1c: Indexing --- 
            # Build a FAISS index for efficient similarity search.
            # print("  Building FAISS search index...") # Uncomment for verbose logging
            try:
                embedding_dim = current_embeddings.shape[1] # Get the dimensionality of the embeddings
                # We use IndexFlatL2, which performs exact search using L2 (Euclidean) distance.
                # For high-dimensional vectors from modern embedding models, cosine similarity often works better,
                # but FAISS's IndexFlatIP (Inner Product) is closely related. For normalized embeddings (like many BGE models),
                # L2 distance and Inner Product/Cosine Similarity ranking are equivalent.
                current_index = faiss.IndexFlatL2(embedding_dim)
                # Add the chunk embeddings to the index. FAISS requires float32 data type.
                current_index.add(current_embeddings.astype('float32'))
                
                if current_index.ntotal == 0:
                     raise ValueError("FAISS index is empty after adding vectors. No vectors were added.")
                # print(f"    FAISS index ready with {current_index.ntotal} vectors.") # Uncomment for verbose logging
            except Exception as e:
                print(f"    ERROR building FAISS index for Size={chunk_size}, Overlap={chunk_overlap}: {e}. Skipping this chunk config.")
                # Reset variables to indicate failure
                last_chunk_size, last_overlap = -1, -1
                current_index = None
                current_embeddings = None
                current_chunks = []
                continue # Skip to the next parameter combination
        
        # --- 8.2 Testing RAG Strategies for the Current Top-K --- 
        # If we reach this point, we have a valid index and chunks for the current chunk_size/overlap.
        
        # Check if the index and chunks are actually available (safety check)
        if current_index is None or not current_chunks:
            print(f"    WARNING: Index or chunks not available for Size={chunk_size}, Overlap={chunk_overlap}. Skipping Top-K={top_k} test.")
            continue
            
        # --- 8.3 Running & Evaluating a Single RAG Strategy --- 
        # Define a nested function to perform the core RAG steps (retrieve, generate, evaluate)
        # This avoids code repetition for each strategy.
        def run_and_evaluate(strategy_name, query_to_use, k_retrieve, use_simulated_rerank=False):
            # print(f"    Starting: {strategy_name} (k={k_retrieve}) ...") # Uncomment for verbose logging
            run_start_time = time.time() # Record start time for timing the run
            
            # Initialize a dictionary to store results for this specific run
            result = {
                'chunk_size': chunk_size, 'overlap': chunk_overlap, 'top_k': k_retrieve, 
                'strategy': strategy_name,
                'retrieved_indices': [], 'rewritten_query': None, 'answer': 'Error: Execution Failed',
                'faithfulness': 0.0, 'relevancy': 0.0, 'similarity_score': 0.0, 'avg_score': 0.0, 
                'time_sec': 0.0
            }
            # Store the rewritten query if applicable
            if strategy_name == "Query Rewrite RAG": 
                result['rewritten_query'] = query_to_use

            try:
                # --- Retrieval Step --- 
                k_for_search = k_retrieve # Number of chunks to retrieve initially
                if use_simulated_rerank:
                    # For simulated rerank, retrieve more candidates initially
                    k_for_search = k_retrieve * RERANK_RETRIEVAL_MULTIPLIER
                    # print(f"      Rerank: Retrieving initial {k_for_search} candidates.") # Uncomment for verbose logging
                
                # 1. Embed the query (original or rewritten)
                query_embedding_response = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=[query_to_use])
                query_embedding = query_embedding_response.data[0].embedding
                query_vector = np.array([query_embedding]).astype('float32') # FAISS needs float32 numpy array
                
                # 2. Perform the search in the FAISS index
                # Ensure k is not greater than the total number of items in the index
                actual_k = min(k_for_search, current_index.ntotal)
                if actual_k == 0:
                    raise ValueError("Index is empty or k_for_search is zero, cannot search.")
                
                # `current_index.search` returns distances and indices of the nearest neighbors
                distances, indices = current_index.search(query_vector, actual_k)
                
                # 3. Process retrieved indices
                # Indices can contain -1 if fewer than 'actual_k' vectors are found (shouldn't happen with IndexFlatL2 unless k > ntotal)
                retrieved_indices_all = indices[0]
                valid_indices = retrieved_indices_all[retrieved_indices_all != -1].tolist()
                
                # 4. Apply simulated reranking (if applicable)
                # In this simulation, we just take the top 'k_retrieve' results from the initially retrieved set.
                # A real reranker would re-score these 'k_for_search' candidates based on relevance to the query.
                if use_simulated_rerank:
                    final_indices = valid_indices[:k_retrieve]
                    # print(f"      Rerank: Selected top {len(final_indices)} indices after simulated rerank.") # Uncomment for verbose logging
                else:
                    final_indices = valid_indices # Use all valid retrieved indices up to k_retrieve
                
                result['retrieved_indices'] = final_indices
                
                # 5. Get the actual text chunks corresponding to the final indices
                retrieved_chunks = [current_chunks[i] for i in final_indices]
                
                # Handle case where no chunks were retrieved (should be rare with valid indices)
                if not retrieved_chunks:
                    print(f"      Warning: No relevant chunks found for {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}). Setting answer to indicate this.")
                    result['answer'] = "No relevant context found in the documents based on the query."
                    # Keep scores at 0.0 as no answer was generated from context
                else:
                    # --- Generation Step --- 
                    # Combine the retrieved chunks into a single context string
                    context_str = "\n\n".join(retrieved_chunks)
                    
                    # Define the system prompt for the generation LLM
                    sys_prompt_gen = "You are a helpful AI assistant. Answer the user's query based strictly on the provided context. If the context doesn't contain the answer, state that clearly. Be concise."
                    
                    # Construct the user prompt including the context and the *original* query
                    # It's important to use the original query here for generating the final answer, even if a rewritten query was used for retrieval.
                    user_prompt_gen = f"Context:\n------\n{context_str}\n------\n\nQuery: {test_query}\n\nAnswer:"
                    
                    # Make the API call to the Nebius generation model
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
                    # Extract the generated text answer
                    generated_answer = gen_response.choices[0].message.content.strip()
                    result['answer'] = generated_answer
                    # Optional: print a snippet of the generated answer
                    # print(f"      Generated Answer: {generated_answer[:100].replace('\n', ' ')}...") 

                    # --- Evaluation Step --- 
                    # Evaluate the generated answer using Faithfulness, Relevancy, and Similarity
                    # print(f"      Evaluating answer... (Faithfulness, Relevancy, Similarity)") # Uncomment for verbose logging
                    
                    # Prepare parameters for evaluation calls (use low temperature for deterministic scoring)
                    eval_params = {'model': NEBIUS_EVALUATION_MODEL, 'temperature': 0.0, 'max_tokens': 10}
                    
                    # 1. Faithfulness Evaluation Call
                    prompt_f = FAITHFULNESS_PROMPT.format(question=test_query, response=generated_answer, true_answer=true_answer_for_query)
                    try:
                        resp_f = client.chat.completions.create(messages=[{"role": "user", "content": prompt_f}], **eval_params)
                        # Attempt to parse the score, clamp between 0.0 and 1.0
                        result['faithfulness'] = max(0.0, min(1.0, float(resp_f.choices[0].message.content.strip())))
                    except Exception as eval_e:
                        print(f"      Warning: Faithfulness score parsing error for {strategy_name} - {eval_e}. Score set to 0.0")
                        result['faithfulness'] = 0.0

                    # 2. Relevancy Evaluation Call
                    prompt_r = RELEVANCY_PROMPT.format(question=test_query, response=generated_answer)
                    try:
                        resp_r = client.chat.completions.create(messages=[{"role": "user", "content": prompt_r}], **eval_params)
                        # Attempt to parse the score, clamp between 0.0 and 1.0
                        result['relevancy'] = max(0.0, min(1.0, float(resp_r.choices[0].message.content.strip())))
                    except Exception as eval_e:
                        print(f"      Warning: Relevancy score parsing error for {strategy_name} - {eval_e}. Score set to 0.0")
                        result['relevancy'] = 0.0
                    
                    # 3. Similarity Score Calculation
                    result['similarity_score'] = calculate_cosine_similarity(
                        generated_answer, 
                        true_answer_for_query, 
                        client, 
                        NEBIUS_EMBEDDING_MODEL
                    )
                    
                    # 4. Calculate Average Score (Faithfulness, Relevancy, Similarity)
                    result['avg_score'] = (result['faithfulness'] + result['relevancy'] + result['similarity_score']) / 3.0
            
            except Exception as e:
                # Catch any unexpected errors during the retrieve/generate/evaluate process
                error_message = f"ERROR during {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}): {str(e)[:200]}..."
                print(f"    {error_message}")
                result['answer'] = error_message # Store the error in the answer field
                # Ensure scores remain at their default error state (0.0)
                result['faithfulness'] = 0.0
                result['relevancy'] = 0.0
                result['similarity_score'] = 0.0
                result['avg_score'] = 0.0
            
            # Record the total time taken for this run
            run_end_time = time.time()
            result['time_sec'] = run_end_time - run_start_time
            
            # Print a summary line for this run (useful for monitoring progress)
            print(f"    Finished: {strategy_name} (C={chunk_size}, O={chunk_overlap}, K={k_retrieve}). AvgScore={result['avg_score']:.2f}, Time={result['time_sec']:.2f}s")
            return result
        # --- End of run_and_evaluate nested function ---

        # --- Execute the RAG Strategies using the run_and_evaluate function --- 
        
        # Strategy 1: Simple RAG (Use original query for retrieval)
        result_simple = run_and_evaluate("Simple RAG", test_query, top_k)
        all_results.append(result_simple)

        # Strategy 2: Query Rewrite RAG 
        rewritten_q = test_query # Default to original query if rewrite fails
        try:
             # print("    Attempting query rewrite for Rewrite RAG...") # Uncomment for verbose logging
             # Define prompts for the query rewriting task
             sys_prompt_rw = "You are an expert query optimizer. Rewrite the user's query to be ideal for vector database retrieval. Focus on key entities, concepts, and relationships. Remove conversational fluff. Output ONLY the rewritten query text."
             user_prompt_rw = f"Original Query: {test_query}\n\nRewritten Query:"
             
             # Call the LLM to rewrite the query
             resp_rw = client.chat.completions.create(
                 model=NEBIUS_GENERATION_MODEL, # Can use the generation model for this task too
                 messages=[
                     {"role": "system", "content": sys_prompt_rw},
                     {"role": "user", "content": user_prompt_rw}
                 ],
                 temperature=0.1, # Low temp for focused rewrite
                 max_tokens=100, 
                 top_p=0.9
             )
             # Clean up the LLM's response to get just the query text
             candidate_q = resp_rw.choices[0].message.content.strip()
             # Remove potential prefixes like "Rewritten Query:" or "Query:"
             candidate_q = re.sub(r'^(rewritten query:|query:)\s*', '', candidate_q, flags=re.IGNORECASE).strip('"')
             
             # Use the rewritten query only if it's reasonably different and not too short
             if candidate_q and len(candidate_q) > 5 and candidate_q.lower() != test_query.lower(): 
                 rewritten_q = candidate_q
                 # print(f"      Using rewritten query: '{rewritten_q}'") # Uncomment for verbose logging
             # else: 
                 # print("      Rewrite failed, too short, or same as original. Using original query.") # Uncomment for verbose logging
        except Exception as e:
             print(f"    Warning: Error during query rewrite: {e}. Using original query.")
             rewritten_q = test_query # Fallback to original query on error
             
        # Run evaluation using the (potentially) rewritten query for retrieval
        result_rewrite = run_and_evaluate("Query Rewrite RAG", rewritten_q, top_k)
        all_results.append(result_rewrite)

        # Strategy 3: Rerank RAG (Simulated)
        # Use original query for retrieval, but simulate the reranking process
        result_rerank = run_and_evaluate("Rerank RAG (Simulated)", test_query, top_k, use_simulated_rerank=True)
        all_results.append(result_rerank)

    print("\n=== RAG Experiment Loop Finished ===")
    print("-" * 25)
```

### 9. Analysis: Reviewing the Results

Now that the experiment loop has completed and `all_results` contains the data from each run, we'll use the Pandas library to analyze the findings.

1.  **Create DataFrame:** Convert the list of result dictionaries (`all_results`) into a Pandas DataFrame for easy manipulation and viewing.
2.  **Sort Results:** Sort the DataFrame by the `avg_score` (the average of Faithfulness, Relevancy, and Similarity) in descending order, so the best-performing configurations appear first.
3.  **Display Top Configurations:** Show the top N rows of the sorted DataFrame, including key parameters, scores, and the generated answer, to quickly identify promising settings.
4.  **Summarize Best Run:** Print a clear summary of the single best-performing configuration based on the average score, showing its parameters, individual scores, time taken, and the full answer it generated.

```python
print("--- Analyzing Experiment Results ---")

# First, check if any results were actually collected
if not all_results:
    print("No results were generated during the experiment. Cannot perform analysis.")
else:
    # Convert the list of result dictionaries into a Pandas DataFrame
    results_df = pd.DataFrame(all_results)
    print(f"Total results collected: {len(results_df)}")
    
    # Sort the DataFrame based on the 'avg_score' column in descending order (best first)
    # Use reset_index(drop=True) to get a clean 0-based index after sorting.
    results_df_sorted = results_df.sort_values(by='avg_score', ascending=False).reset_index(drop=True)
    
    print("\n--- Top 10 Performing Configurations (Sorted by Average Score) ---")
    # Define the columns we want to display in the summary table
    display_cols = [
        'chunk_size', 'overlap', 'top_k', 'strategy', 
        'avg_score', 'faithfulness', 'relevancy', 'similarity_score', # Added similarity
        'time_sec', 
        'answer' # Including the answer helps qualitatively assess the best runs
    ]
    # Filter out any columns that might not exist (e.g., if an error occurred before population)
    display_cols = [col for col in display_cols if col in results_df_sorted.columns]
    
    # Display the head (top 10 rows) of the sorted DataFrame using the selected columns
    # The display() function provides richer output in Jupyter environments.
    display(results_df_sorted[display_cols].head(10))
    
    # --- Summary of the Single Best Run --- 
    print("\n--- Best Configuration Summary ---")
    # Check if the sorted DataFrame is not empty before accessing the first row
    if not results_df_sorted.empty:
        # Get the first row (index 0), which corresponds to the best score after sorting
        best_run = results_df_sorted.iloc[0]
        
        # Print the parameters and results of the best configuration
        print(f"Chunk Size: {best_run.get('chunk_size', 'N/A')} words")
        print(f"Overlap: {best_run.get('overlap', 'N/A')} words")
        print(f"Top-K Retrieved: {best_run.get('top_k', 'N/A')} chunks")
        print(f"Strategy: {best_run.get('strategy', 'N/A')}")
        # Use .get(col, default) for robustness in case a column is missing
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
        # Print the full answer generated by the best configuration
        print(best_answer)
    else:
        # Handle the case where no results were successfully processed
        print("Could not determine the best configuration (no valid results found).")
        
print("\n--- Analysis Complete --- ")
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
