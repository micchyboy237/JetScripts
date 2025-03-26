import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = sqlite3.connect("data/anime.db")
cursor = conn.cursor()
cursor.execute("SELECT title, synopsis FROM anime")
data = cursor.fetchall()

texts = [f"{title} - {synopsis}" for title, synopsis in data]
embeddings = np.array(model.encode(texts))

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "data/faiss_index/anime.index")

conn.close()
