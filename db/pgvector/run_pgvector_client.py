import numpy as np
from jet.db.pgvector import PgVectorClient

# Database configuration
TABLE_NAME = "embeddings"
VECTOR_DIM = 3

with PgVectorClient(dbname="jobs_db1", user="jethroestrada", password="", host="localhost", port=5432) as client:
    client.create_table(TABLE_NAME, VECTOR_DIM)

    try:
        # Insert a single vector with a hash ID
        vector = np.random.rand(VECTOR_DIM).tolist()
        vector_id = client.insert_vector(TABLE_NAME, vector)
        print(f"Inserted vector ID: {vector_id}")

        # Retrieve the inserted vector by ID
        retrieved_vector = client.get_vector_by_id(TABLE_NAME, vector_id)
        print(f"Retrieved vector: {retrieved_vector}")

        # Insert multiple vectors using batch insertion
        vectors = [np.random.rand(VECTOR_DIM).tolist() for _ in range(3)]
        vector_ids = client.insert_vectors(TABLE_NAME, vectors)
        print(f"Inserted multiple vector IDs: {vector_ids}")

        # Retrieve multiple vectors by their IDs
        retrieved_vectors = client.get_vectors_by_ids(TABLE_NAME, vector_ids)
        print(f"Retrieved vectors: {retrieved_vectors}")

        # Insert a vector with a specific predefined ID
        specific_id = "custom-123"
        specific_vector = np.random.rand(VECTOR_DIM).tolist()
        client.insert_vector_by_id(TABLE_NAME, specific_id, specific_vector)
        print(f"Inserted vector with specific ID: {specific_id}")

        # Retrieve the vector using the specific ID
        retrieved_specific_vector = client.get_vector_by_id(
            TABLE_NAME, specific_id)
        print(
            f"Retrieved vector for ID {specific_id}: {retrieved_specific_vector}")

        # Insert multiple vectors with predefined IDs
        specific_vectors = {
            "vec-1": np.random.rand(VECTOR_DIM).tolist(),
            "vec-2": np.random.rand(VECTOR_DIM).tolist(),
            "vec-3": np.random.rand(VECTOR_DIM).tolist(),
        }
        client.insert_vector_by_ids(TABLE_NAME, specific_vectors)
        print(
            f"Inserted multiple vectors with specific IDs: {list(specific_vectors.keys())}")

        # Retrieve the inserted specific vectors
        retrieved_specific_vectors = client.get_vectors_by_ids(
            TABLE_NAME, list(specific_vectors.keys()))
        print(f"Retrieved specific vectors: {retrieved_specific_vectors}")

        # Update a single vector
        new_vector = np.random.rand(VECTOR_DIM).tolist()
        client.update_vector_by_id(TABLE_NAME, vector_id, new_vector)
        print(f"Updated vector ID {vector_id}")

        # Update multiple vectors at once
        updates = {vid: np.random.rand(VECTOR_DIM).tolist()
                   for vid in vector_ids}
        client.update_vector_by_ids(TABLE_NAME, updates)
        print(f"Updated vectors with IDs: {vector_ids}")

        # Search for top 3 most similar vectors
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        similar_vectors = client.search_similar(
            TABLE_NAME, query_vector, top_k=3)
        print(f"Top 3 similar vectors: {similar_vectors}")

        # Cleanup: Drop all rows in the table
        client.drop_all_rows(TABLE_NAME)
        print("Deleted all rows.")

        # Cleanup: Drop all tables in the database
        client.delete_all_tables()
        print("Deleted all tables.")

    except Exception as e:
        print(f"Transaction failed: {e}")
