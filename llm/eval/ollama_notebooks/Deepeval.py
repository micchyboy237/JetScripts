from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.base import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
```python
# !pip install -q -q llama-index
# !pip install -U -q deepeval
# !deepeval login


def main():
    # Initialize ollama settings
    initialize_ollama_settings()

    # Replace openai llm and embed models with ollama
    from jet.llm.ollama.base import Ollama
    model = "llama3.1"
    from jet.llm.ollama.base import OllamaEmbedding
    embedding = OllamaEmbedding(
        model_name="nomic-embed-text", base_url="http://localhost:11434")

    # Load data and create index
    documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    rag_application = index.as_query_engine()

    # Evaluate response using DeepEvalFaithfulnessEvaluator
    user_input = "What is LlamaIndex?"
    response_object = rag_application.query(user_input)
    evaluator = DeepEvalFaithfulnessEvaluator()
    evaluation_result = evaluator.evaluate_response(
        query=user_input, response=response_object
    )
    print(evaluation_result)


if __name__ == "__main__":
    main()
```
