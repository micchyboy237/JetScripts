import os
import json
import numpy as np
import pandas as pd
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from jet.llm import get_model_path

model_path = get_model_path("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(model_path)

client = QdrantClient("http://jetairm1:6333")

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_file = os.path.join(base_dir, "startups_demo.json")
vectors_file = os.path.join(base_dir, "startup_vectors.npy")


def prepare_dataset():
    df = pd.read_json(dataset_file, lines=True)

    vectors = model.encode(
        [row.alt + ". " + row.description for row in df.itertuples()],
        show_progress_bar=True,
    )

    # All of the descriptions are now converted into vectors. There are 40474 vectors of 384 dimensions. The output layer of the model has this dimension

    vectors.shape
    # > (40474, 384)

    # Download the saved vectors into a new file named startup_vectors.npy
    np.save(vectors_file, vectors, allow_pickle=False)


def upload_data():
    # Related vectors need to be added to a collection. Create a new collection for your startup vectors.
    if not client.collection_exists("startups"):
        client.create_collection(
            collection_name="startups",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # The Qdrant client library defines a special function that allows you to load datasets into the service. However, since there may be too much data to fit a single computer memory, the function takes an iterator over the data as input.

    fd = open(dataset_file)

    # payload is now an iterator over startup data
    payload = map(json.loads, fd)

    # Load all vectors into memory, numpy array works as iterable for itself.
    # Other option would be to use Mmap, if you don't want to load all data into RAM
    vectors = np.load(vectors_file)

    # Upload the data
    client.upload_collection(
        collection_name="startups",
        vectors=vectors,
        payload=payload,
        ids=None,  # Vector ids will be assigned automatically
        batch_size=256,  # How many vectors will be uploaded in a single request?
    )

    # Vectors are now uploaded to Qdrant.


def setup():
    prepare_dataset()
    upload_data()


if __name__ == '__main__':
    setup()
