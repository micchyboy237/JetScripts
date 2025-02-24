from jet.llm.ollama.embeddings import get_ollama_embedding_function


if __name__ == '__main__':
    texts = [
        'Text 1'
    ]
    embedding_function = get_ollama_embedding_function(
        model="nomic-embed-text"
    )
    embedding_function(texts)
