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

**Remember!** After the installation finishes, you might need to **Restart the Kernel** (or Runtime) for Jupyter/Colab to recognize the newly installed packages. Look for this option in the menu (e.g., 'Kernel' -> 'Restart Kernel...' or 'Runtime' -> 'Restart Runtime').

### 2. Setup: Importing Libraries

With the libraries installed, we import them into our Python environment to make their functions available.

### 3. Configuration: Setting Up Our Experiment

Here, we define all the settings and parameters for our experiment directly as Python variables. This makes it easy to see and modify the configuration in one place.

**Key Configuration Areas:**
*   **Nebius API Details:** Credentials and model identifiers for connecting to Nebius AI.
*   **LLM Settings:** Parameters controlling the behavior of the language model during answer generation (e.g., `temperature` for creativity).
*   **Evaluation Prompts:** The specific instructions (prompts) given to the LLM when it acts as an evaluator for Faithfulness and Relevancy.
*   **Tuning Parameters:** The different values for chunk size, overlap, and retrieval `top_k` that we want to systematically test.
*   **Reranking Setting:** Configuration for the simulated reranking strategy.

### 4. Input Data: The Knowledge Source & Our Question

Every RAG system needs a knowledge base to draw information from. Here, we define:
*   `corpus_texts`: A list of strings, where each string is a document containing information (in this case, about renewable energy sources).
*   `test_query`: The specific question we want the RAG system to answer using the `corpus_texts`.
*   `true_answer_for_query`: A carefully crafted 'ground truth' answer based *only* on the information available in `corpus_texts`. This is essential for evaluating Faithfulness and Semantic Similarity accurately.

### 5. Core Component: Text Chunking Function

LLMs and embedding models have limits on the amount of text they can process at once. Furthermore, retrieval works best when searching over smaller, focused pieces of text rather than entire large documents.

**Chunking** is the process of splitting large documents into smaller, potentially overlapping, segments.

- **`chunk_size`**: Determines the approximate size (here, in words) of each chunk.
- **`chunk_overlap`**: Specifies how many words from the end of one chunk should also be included at the beginning of the next chunk. This helps prevent relevant information from being lost if it spans across the boundary between two chunks.

We define a function `chunk_text` to perform this splitting based on word counts.

### 6. Core Component: Connecting to Nebius AI

To use the Nebius AI models (for embedding, generation, evaluation), we need to establish a connection to their API. We use the `openai` Python library, which provides a convenient way to interact with OpenAI-compatible APIs like Nebius.

We instantiate an `OpenAI` client object, providing our API key and the specific Nebius API endpoint URL.

### 7. Core Component: Cosine Similarity Function

To evaluate how semantically similar the generated answer is to our ground truth answer, we use **Cosine Similarity**. This metric measures the cosine of the angle between two vectors (in our case, the embedding vectors of the two answers).

- A score of **1** means the vectors point in the same direction (maximum similarity).
- A score of **0** means the vectors are orthogonal (no similarity).
- A score of **-1** means the vectors point in opposite directions (maximum dissimilarity).

For text embeddings, scores typically range from 0 to 1, where higher values indicate greater semantic similarity.

We define a function `calculate_cosine_similarity` that takes two text strings, generates their embeddings using the Nebius client, and returns their cosine similarity score.

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

### 9. Analysis: Reviewing the Results

Now that the experiment loop has completed and `all_results` contains the data from each run, we'll use the Pandas library to analyze the findings.

1.  **Create DataFrame:** Convert the list of result dictionaries (`all_results`) into a Pandas DataFrame for easy manipulation and viewing.
2.  **Sort Results:** Sort the DataFrame by the `avg_score` (the average of Faithfulness, Relevancy, and Similarity) in descending order, so the best-performing configurations appear first.
3.  **Display Top Configurations:** Show the top N rows of the sorted DataFrame, including key parameters, scores, and the generated answer, to quickly identify promising settings.
4.  **Summarize Best Run:** Print a clear summary of the single best-performing configuration based on the average score, showing its parameters, individual scores, time taken, and the full answer it generated.

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
