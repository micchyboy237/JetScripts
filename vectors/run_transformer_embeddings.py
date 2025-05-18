from jet.llm.utils.transformer_embeddings import generate_embeddings, get_embedding_function, search_docs
from jet.llm.mlx.models import EmbedModelType
from jet.transformers.formatters import format_json
from jet.logger import logger

if __name__ == '__main__':
    documents = [
        "The sky is blue today.",
        "The grass is green.",
        "The sun is shining brightly.",
        "Clouds are white and fluffy."
    ]
    query = "blue sky"
    top_k = 4
    embed_model: EmbedModelType = "all-minilm:33m"

    query_embedding = generate_embeddings(embed_model, query)
    doc_embedding = generate_embeddings(embed_model, documents)

    results = search_docs(query, documents, embed_model, top_k=top_k)

    logger.newline()
    logger.gray("Results:")
    logger.success(format_json(results))
