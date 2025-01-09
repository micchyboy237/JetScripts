from llama_index.core.embeddings import resolve_embed_model
from jet.llm.ollama.base import Ollama
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation.benchmarks import HotpotQAEvaluator
```python
# %pip install llama-index-llms-openai
# !pip install llama-index

initialize_ollama_settings()

# Refactor with main. Initialize ollama.


def main():
    # Replace openai llm and embed models with ollama.
    llm = Ollama(model="llama3.2")
    embed_model = resolve_embed_model(
        "local:sentence-transformers/all-MiniLM-L6-v2"
    )

    # Update variable names appropriately since ollama doesn't have gpt models.
    index = VectorStoreIndex.from_documents(
        [Document.example()], embed_model=embed_model, show_progress=True
    )
    engine = index.as_query_engine(llm=llm)

    HotpotQAEvaluator().run(engine, queries=5, show_result=True)

    # Separate usage examples as functions that will be called in main.
    def rerank():
        from llama_index.core.postprocessor import SentenceTransformerRerank

        rerank = SentenceTransformerRerank(top_n=3)

        engine = index.as_query_engine(
            llm=llm,
            node_postprocessors=[rerank],
        )

        HotpotQAEvaluator().run(engine, queries=5, show_result=True)

    # Call the rerank function in main.
    rerank()


# Add a main function to contain all usage examples.
if __name__ == "__main__":
    main()
```
