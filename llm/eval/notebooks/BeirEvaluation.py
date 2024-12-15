%pip install llama-index-embeddings-huggingface!pip install llama-indexfrom llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation.benchmarks import BeirEvaluator
from llama_index.core import VectorStoreIndex


def create_retriever(documents):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, show_progress=True
    )
    return index.as_retriever(similarity_top_k=30)


BeirEvaluator().run(
    create_retriever, datasets=["nfcorpus"], metrics_k_values=[3, 10, 30]
)
