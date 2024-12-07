# %% [markdown]
# # How RAG with txtai works
#
# [txtai](https://github.com/neuml/txtai) is an all-in-one embeddings database for semantic search, LLM orchestration and language model workflows.

# %% [markdown]
# # Install dependencies
#
# Install `txtai` and all dependencies.

# %%
import json
import os
import traceback
from txtai.vectors import VectorsFactory
from txtai import LLM, RAG
from jet.llm.embeddings import Embeddings
from jet.logger import logger
# % % capture
# egg=txtai[api,pipeline] autoawq
# !pip install git+https: // github.com/neuml/txtai

# %% [markdown]
# # Components of a RAG pipeline

# %%
from jet.llm.search import (
    ScoringMethod,
    load_local_json,
    load_or_create_embeddings,
    build_ann_index,
    ann_search,
    scoring_search,
)
from jet.logger import logger

GENERATED_DIR = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/evaluation/examples/txtai/generated"
# Cache directory for embeddings
EMBEDDINGS_DIR = f"{GENERATED_DIR}/embeddings"
RESULTS_DIR = f"{GENERATED_DIR}/search"
EMBEDDINGS_CACHE_KEY = "crew_ai_docs"


def create_llm():
    """Create LLM instance."""
    return LLM(path="ollama/llama3.1", method="litellm", api_base="http://localhost:11434")


def generate_prompt(question, context):
    """Generate prompt with question and context."""
    return f"""<|im_start|>system
You are a friendly assistant. You answer questions from users.<|im_end|>
<|im_start|>user
Answer the following question using only the context below. Only include information
specifically discussed.

question: {question}
context: {context} <|im_end|>
<|im_start|>assistant
"""


def get_context(embeddings, question):
    """Generate context based on question using embeddings."""
    return "\n".join([x["text"] for x in embeddings.search(question)])


def run_rag_pipeline(question, embeddings, llm, prompt, texts: list[str]):
    """Run the RAG pipeline to get the answer."""
    rag = RAG(embeddings, llm, template=prompt)
    rag_results = rag(
        question,
        maxlength=8192,
        texts=texts,
        # messages=[
        #     {"content": question, "role": "user"}],
        stream=True
    )
    return rag_results


def save_results(results, filename):
    """Save results to a JSON file."""
    results_path = os.path.join(RESULTS_DIR, filename)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.log("Saved to:", results_path, colors=[
               "LOG", "BRIGHT_SUCCESS"])


def main():
    embedding_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    dataset_path = "/Users/jethroestrada/Desktop/External_Projects/AI/agents_2/crewAI/my_project/src/my_project/generated/rag/crewai-docs.json"
    dataset = load_local_json(dataset_path)
    texts = [row["page_content"] for row in dataset]
    embeddings_model = VectorsFactory.create({"path": embedding_model}, None)
    embeddings_dir = os.path.join(EMBEDDINGS_DIR, EMBEDDINGS_CACHE_KEY)
    cache_file = os.path.join(embeddings_dir, "embeddings.npy")
    # embeddings = Embeddings()
    # embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
    embeddings = load_or_create_embeddings(
        texts, model=embeddings_model, cache_file=cache_file)

    # Generate context
    # context = get_context(embeddings, question)
    query = "crewai setup"
    top_k = 5
    max_context_length = 800

    ann = build_ann_index(embeddings)
    ann_results = ann_search(
        query, ann, model=embeddings_model, dataset=dataset, top_k=top_k)
    contexts = [
        f"Tags: {", ".join(item['tags'])}" + "\n" +
        item['text']
        for item in ann_results
    ]
    context = "\n\n".join(contexts)
    # context = context[0: max_context_length]

    llm = create_llm()

    # Question to ask
    question = "How do create agents with tasks?"

    # Generate prompt
    prompt = generate_prompt(question, context)

    # Run RAG pipeline
    # Example setup (you would need to initialize the similarity model, tokenizer, etc.)
    # Assuming you have a similarity model
    from txtai.pipeline import Tokenizer, Similarity
    similarity_instance = Similarity(path=embedding_model)
    # answer = run_rag_pipeline(question, embeddings, llm, prompt)
    chunks = run_rag_pipeline(
        question, similarity_instance, llm, prompt, texts=texts)

    answer = ""
    for chunk in chunks:
        answer += chunk
        logger.success(chunk, flush=True)

    # Save answer
    save_results({
        "type": "rag",
        "method": "question",
        "question": question,
        "context": context,
        "prompt": prompt,
        "response": answer
    }, f"rag_scores.json")


# Run the main function
if __name__ == "__main__":
    main()
