from jet.llm.utils.transformer_embeddings import generate_embeddings, get_embedding_function, search_docs
from jet.llm.mlx.models import EmbedModelType
from jet.transformers.formatters import format_json
from jet.logger import logger


if __name__ == "__main__":
    from jet.file.utils import load_file, save_file
    import os

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    headers: list[dict] = load_file(docs_file)
    documents = [header["text"] for header in headers]
    query = f"List trending isekai reincarnation anime this year."

    top_k = 10
    embed_model: EmbedModelType = "all-minilm:33m"

    query_embedding = generate_embeddings(embed_model, query)
    doc_embedding = generate_embeddings(embed_model, documents)

    results = search_docs(query, documents, embed_model, top_k=top_k)

    logger.newline()
    logger.gray("Results:")
    logger.success(format_json(results))

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(results, f"{output_dir}/results.json")
