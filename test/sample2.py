from typing import Optional, Union
from llama_index.memory.mem0.base import BaseMem0, Mem0Context
from jet.db.chroma import ChromaClient, VectorItem, SearchResult, GetResult


class Mem0Memory(BaseMem0):
    chroma_client: Optional[ChromaClient] = None

    def __init__(
        self,
        chroma_client: Optional[ChromaClient] = None,
        context: Optional[Mem0Context] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if chroma_client is not None:
            self.chroma_client = chroma_client
        if context is not None:
            self.context = context

    def add(
        self, messages: Union[str, list[dict[str, str]]], collection_name: str, **kwargs
    ) -> None:
        """Add messages to Chroma collection."""
        if not self.chroma_client:
            raise ValueError("ChromaClient is not initialized")

        vector_items = [
            VectorItem(
                id=str(i),
                text=msg["content"],
                # Assume vector embedding passed in kwargs
                vector=kwargs.get("vector"),
                metadata={"role": msg["role"]},
            )
            for i, msg in enumerate(messages)
        ]

        self.chroma_client.insert(
            collection_name=collection_name, items=vector_items)

    def search(
        self, query_vector: list[float], collection_name: str, limit: int = 5
    ) -> Optional[SearchResult]:
        """Search messages in Chroma collection."""
        if not self.chroma_client:
            raise ValueError("ChromaClient is not initialized")

        return self.chroma_client.search(
            collection_name=collection_name, vectors=[
                query_vector], limit=limit
        )

    def reset(self, collection_name: str) -> None:
        """Reset Chroma collection."""
        if not self.chroma_client:
            raise ValueError("ChromaClient is not initialized")

        self.chroma_client.delete_collection(collection_name=collection_name)


if __name__ == "__main__":
    from jet.llm.ollama import initialize_ollama_settings, get_embedding_function
    initialize_ollama_settings()

    embedding_func = get_embedding_function(
        embedding_model="mxbai-embed-large"
    )

    # Initialize ChromaClient
    chroma_client = ChromaClient(data_path="data/vector_db")

    # Initialize Mem0Memory
    memory = Mem0Memory(
        chroma_client=chroma_client,
        context={"user_id": "12345"},
    )

    # Add messages
    memory.add(
        messages=[
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm great, thank you!"},
        ],
        collection_name="chat_memory",
        vector=[0.1, 0.2, 0.3],  # Example vector
    )

    # Search messages
    search_result = memory.search(
        query_vector=[0.1, 0.2, 0.3], collection_name="chat_memory", limit=5
    )
    print(search_result)

    # Reset memory
    memory.reset(collection_name="chat_memory")
