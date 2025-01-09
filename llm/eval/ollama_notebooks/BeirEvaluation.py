from jet.llm.ollama.base import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation.benchmarks import BeirEvaluator
from jet.llm.ollama.base import OllamaEmbedding
```python
# %pip install llama-index-embeddings-huggingface
# !pip install llama-index


def initialize_ollama_settings():
    # Initialize ollama settings
    pass


initialize_ollama_settings()

# Replace openai llm and embed models with ollama


def create_retriever(documents):
    # Initialize ollama settings
    initialize_ollama_settings()

    # Replace openai embed model with ollama embed model
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text", base_url="http://localhost:11434")

    # Create vector store index from documents using ollama embed model
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, show_progress=True
    )

    # Return retriever object
    return index.as_retriever(similarity_top_k=30)


# Run beir evaluator with ollama retriever
BeirEvaluator().run(create_retriever, datasets=[
    "nfcorpus"], metrics_k_values=[3, 10, 30])
```
