from jet.logger import logger
import numpy as np
from jet.db.postgres.pgvector import PgVectorClient

# Database configuration
TABLE_NAME = "embeddings"
VECTOR_DIM = 3

with PgVectorClient(
    dbname="vector_db1"
) as client:
    try:
        # Clear data
        client.delete_all_tables()

        client.create_table(TABLE_NAME, VECTOR_DIM)

        # Insert a single vector with a hash ID
        vector = np.random.rand(VECTOR_DIM).tolist()
        vector_id = client.insert_vector(TABLE_NAME, vector)
        logger.newline()
        logger.debug(f"Inserted vector ID:")
        logger.success(f"{vector_id}")

        # Retrieve the inserted vector by ID
        retrieved_vector = client.get_vector_by_id(TABLE_NAME, vector_id)
        logger.newline()
        logger.debug(f"Retrieved vector:")
        logger.success(f"{retrieved_vector}")

        # Insert multiple vectors using batch insertion
        vectors = [np.random.rand(VECTOR_DIM).tolist() for _ in range(3)]
        vector_ids = client.insert_vectors(TABLE_NAME, vectors)
        logger.newline()
        logger.debug(f"Inserted multiple vector IDs:")
        logger.success(f"{vector_ids}")

        # Retrieve multiple vectors by their IDs
        retrieved_vectors = client.get_vectors_by_ids(TABLE_NAME, vector_ids)
        logger.newline()
        logger.debug(f"Retrieved vectors:")
        logger.success(f"{retrieved_vectors}")

        # Retrieve all vectors in the table
        all_vectors = client.get_vectors(TABLE_NAME)
        logger.newline()
        logger.debug(f"All vectors in the table:")
        logger.success(f"{all_vectors}")

        # Count the total number of vectors in the table
        vector_count = client.count_vectors(TABLE_NAME)
        logger.newline()
        logger.debug(f"Total number of vectors:")
        logger.success(f"{vector_count}")

        # Insert a vector with a specific predefined ID
        specific_id = "custom-123"
        specific_vector = np.random.rand(VECTOR_DIM).tolist()
        client.insert_vector_by_id(TABLE_NAME, specific_id, specific_vector)
        logger.newline()
        logger.debug(f"Inserted vector with specific ID:")
        logger.success(f"{specific_id}")

        # Retrieve the vector using the specific ID
        retrieved_specific_vector = client.get_vector_by_id(
            TABLE_NAME, specific_id)
        logger.newline()
        logger.debug(f"Retrieved vector for ID {specific_id}:")
        logger.success(f"{retrieved_specific_vector}")

        # Insert multiple vectors with predefined IDs
        specific_vectors = {
            "vec-1": np.random.rand(VECTOR_DIM).tolist(),
            "vec-2": np.random.rand(VECTOR_DIM).tolist(),
            "vec-3": np.random.rand(VECTOR_DIM).tolist(),
        }
        client.insert_vector_by_ids(TABLE_NAME, specific_vectors)
        logger.newline()
        logger.debug(f"Inserted multiple vectors with specific IDs:")
        logger.success(list(specific_vectors.keys()))

        # Retrieve the inserted specific vectors
        retrieved_specific_vectors = client.get_vectors_by_ids(
            TABLE_NAME, list(specific_vectors.keys()))
        logger.newline()
        logger.debug(f"Retrieved specific vectors:")
        logger.success(retrieved_specific_vectors)

        # Update a single vector
        new_vector = np.random.rand(VECTOR_DIM).tolist()
        client.update_vector_by_id(TABLE_NAME, vector_id, new_vector)
        logger.newline()
        logger.debug(f"Updated vector ID:")
        logger.success(f"{vector_id}")

        # Update multiple vectors at once
        updates = {vid: np.random.rand(VECTOR_DIM).tolist()
                   for vid in vector_ids}
        client.update_vector_by_ids(TABLE_NAME, updates)
        logger.newline()
        logger.debug(f"Updated vectors with IDs:")
        logger.success(vector_ids)

        # Search for top 3 most similar vectors
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        similar_vectors = client.search_similar(
            TABLE_NAME, query_vector, top_k=3)
        logger.newline()
        logger.debug(f"Top 3 similar vectors:")
        logger.success(similar_vectors)

        # Cleanup: Drop all rows in the table
        client.drop_all_rows(TABLE_NAME)
        logger.newline()
        logger.warning("Deleted all rows.")

        # Cleanup: Drop all tables in the database
        client.delete_all_tables()
        logger.newline()
        logger.warning("Deleted all tables.")

    except Exception as e:
        logger.newline()
        logger.error(f"Transaction failed:\n{e}")
