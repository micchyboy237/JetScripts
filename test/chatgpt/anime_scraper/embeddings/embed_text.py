import sqlite3
import faiss
from jet.llm.utils.embeddings import get_embedding_function
from jet.token.token_utils import get_model_max_tokens
import openai
import numpy as np
import os

DB_PATH = "data/articles.db"
FAISS_PATH = "data/faiss_index"

embed_model = "mxbai-embed-large"
embed_text = get_embedding_function(embed_model)


def get_texts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM articles")
    data = cursor.fetchall()
    conn.close()
    return data


def store_embeddings():
    data = get_texts()
    dimension = get_model_max_tokens(embed_model)
    index = faiss.IndexFlatL2(dimension)

    ids, vectors = [], []
    for row_id, content in data:
        embedding = embed_text(content)
        vectors.append(embedding)
        ids.append(row_id)

    index.add(np.array(vectors))
    faiss.write_index(index, FAISS_PATH)


if __name__ == "__main__":
    store_embeddings()
