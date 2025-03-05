import numpy as np
from jet.db.pgvector import PgVectorClient

# Database configuration
TABLE_NAME = "embeddings"
VECTOR_DIM = 3

with PgVectorClient(dbname="vector_db1", user="jethroestrada", password="", host="localhost", port=5432) as client:
    client.create_table(TABLE_NAME, VECTOR_DIM)

    try:
        # Insert a vector with a hash ID
        vector = np.random.rand(VECTOR_DIM).tolist()
        vector_id = client.insert_vector(TABLE_NAME, vector)
        print(f"Inserted vector ID: {vector_id}")

        # Retrieve vector by ID
        retrieved_vector = client.get_vector_by_id(TABLE_NAME, vector_id)
        print(f"Retrieved vector: {retrieved_vector}")

        # Insert multiple vectors
        vectors = [np.random.rand(VECTOR_DIM).tolist() for _ in range(3)]
        vector_ids = [client.insert_vector(TABLE_NAME, v) for v in vectors]
        print(f"Inserted vector IDs: {vector_ids}")

        # Retrieve multiple vectors
        retrieved_vectors = client.get_vectors_by_ids(TABLE_NAME, vector_ids)
        print(f"Retrieved vectors: {retrieved_vectors}")

        # Update a vector
        new_vector = np.random.rand(VECTOR_DIM).tolist()
        client.update_vector_by_id(TABLE_NAME, vector_id, new_vector)
        print(f"Updated vector ID {vector_id}")

        # Update multiple vectors
        updates = {vid: np.random.rand(VECTOR_DIM).tolist()
                   for vid in vector_ids}
        client.update_vector_by_ids(TABLE_NAME, updates)
        print(f"Updated vectors with IDs: {vector_ids}")

        # Search for similar vectors
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        similar_vectors = client.search_similar(
            TABLE_NAME, query_vector, top_k=3)
        print(f"Top 3 similar vectors: {similar_vectors}")

        # Cleanup: Drop all rows
        client.drop_all_rows(TABLE_NAME)
        print("Deleted all rows.")

        # Cleanup: Delete all tables
        client.delete_all_tables()
        print("Deleted all tables.")

    except Exception as e:
        print(f"Transaction failed: {e}")
