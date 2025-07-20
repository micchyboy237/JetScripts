import os
import shutil
from jet.file.utils import save_file
from jet.logger import logger
import numpy as np
from jet.db.postgres.pgvector import PgVectorClient

# Database configuration
TABLE_NAME = "embeddings"
VECTOR_DIM = 3

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

with PgVectorClient(
    dbname="test_db1",
    overwrite_db=True,
) as client:
    try:
        # Log database metadata at the start
        db_metadata = client.get_database_metadata()
        logger.newline()
        logger.debug("Database metadata:")
        logger.success(f"{db_metadata}")
        save_file(db_metadata, f"{OUTPUT_DIR}/db_metadata.json")

        # Clear data
        client.delete_all_tables()

        # Log all tables (should be empty after deletion)
        tables = client.get_all_tables()
        logger.newline()
        logger.debug("All tables after deletion:")
        logger.success(f"{tables}")
        save_file(tables, f"{OUTPUT_DIR}/tables_after_deletion.json")

        client.create_table(TABLE_NAME, VECTOR_DIM)

        # Log table metadata after creation
        table_metadata = client.get_table_metadata(TABLE_NAME)
        logger.newline()
        logger.debug(f"Metadata for table {TABLE_NAME}:")
        logger.success(f"{table_metadata}")
        save_file(table_metadata,
                  f"{OUTPUT_DIR}/table_metadata_after_creation.json")

        # Insert a single vector with a hash ID
        vector = np.random.rand(VECTOR_DIM).tolist()
        vector_id = client.insert_vector(TABLE_NAME, vector)
        logger.newline()
        logger.debug(f"Inserted vector ID:")
        logger.success(f"{vector_id}")
        save_file(vector_id, f"{OUTPUT_DIR}/inserted_vector_id.txt")

        # Retrieve the inserted vector by ID
        retrieved_vector = client.get_vector_by_id(TABLE_NAME, vector_id)
        logger.newline()
        logger.debug(f"Retrieved vector:")
        logger.success(f"{retrieved_vector}")
        save_file(retrieved_vector, f"{OUTPUT_DIR}/retrieved_vector.json")

        # Insert multiple vectors using batch insertion
        vectors = [np.random.rand(VECTOR_DIM).tolist() for _ in range(3)]
        vector_ids = client.insert_vectors(TABLE_NAME, vectors)
        logger.newline()
        logger.debug(f"Inserted multiple vector IDs:")
        logger.success(f"{vector_ids}")
        save_file(vector_ids, f"{OUTPUT_DIR}/inserted_vector_ids.json")

        # Retrieve multiple vectors by their IDs
        retrieved_vectors = client.get_vectors_by_ids(TABLE_NAME, vector_ids)
        logger.newline()
        logger.debug(f"Retrieved vectors:")
        logger.success(f"{retrieved_vectors}")
        save_file(retrieved_vectors, f"{OUTPUT_DIR}/retrieved_vectors.json")

        # Retrieve all vectors in the table
        all_vectors = client.get_vectors(TABLE_NAME)
        logger.newline()
        logger.debug(f"All vectors in the table:")
        logger.success(f"{all_vectors}")
        save_file(all_vectors, f"{OUTPUT_DIR}/all_vectors.json")

        # Count the total number of vectors in the table
        vector_count = client.count_vectors(TABLE_NAME)
        logger.newline()
        logger.debug(f"Total number of vectors:")
        logger.success(f"{vector_count}")
        save_file(vector_count, f"{OUTPUT_DIR}/vector_count.json")

        # Log table metadata after insertions
        table_metadata = client.get_table_metadata(TABLE_NAME)
        logger.newline()
        logger.debug(f"Updated metadata for table {TABLE_NAME}:")
        logger.success(f"{table_metadata}")
        save_file(table_metadata,
                  f"{OUTPUT_DIR}/table_metadata_after_inserts.json")

        # Insert a vector with a specific predefined ID
        specific_id = "custom-123"
        specific_vector = np.random.rand(VECTOR_DIM).tolist()
        client.insert_vector_by_id(TABLE_NAME, specific_id, specific_vector)
        logger.newline()
        logger.debug(f"Inserted vector with specific ID:")
        logger.success(f"{specific_id}")
        save_file(specific_id, f"{OUTPUT_DIR}/inserted_specific_id.txt")

        # Retrieve the vector using the specific ID
        retrieved_specific_vector = client.get_vector_by_id(
            TABLE_NAME, specific_id)
        logger.newline()
        logger.debug(f"Retrieved vector for ID {specific_id}:")
        logger.success(f"{retrieved_specific_vector}")
        save_file(retrieved_specific_vector,
                  f"{OUTPUT_DIR}/retrieved_specific_vector.json")

        # Insert multiple vectors with predefined IDs
        specific_vectors = {
            "vec-1": np.random.rand(VECTOR_DIM),
            "vec-2": np.random.rand(VECTOR_DIM),
            "vec-3": np.random.rand(VECTOR_DIM),
        }
        # Convert np.ndarray to list for each value to match VectorInput
        specific_vectors_list = {k: v.tolist() if hasattr(
            v, "tolist") else v for k, v in specific_vectors.items()}
        client.insert_vector_by_ids(TABLE_NAME, specific_vectors_list)
        logger.newline()
        logger.debug(f"Inserted multiple vectors with specific IDs:")
        logger.success(list(specific_vectors.keys()))
        save_file(list(specific_vectors.keys()),
                  f"{OUTPUT_DIR}/inserted_specific_ids.json")

        # Retrieve the inserted specific vectors
        retrieved_specific_vectors = client.get_vectors_by_ids(
            TABLE_NAME, list(specific_vectors.keys()))
        logger.newline()
        logger.debug(f"Retrieved specific vectors:")
        logger.success(retrieved_specific_vectors)
        save_file(retrieved_specific_vectors,
                  f"{OUTPUT_DIR}/retrieved_specific_vectors.json")

        # Update a single vector
        new_vector = np.random.rand(VECTOR_DIM).tolist()
        client.update_vector_by_id(TABLE_NAME, vector_id, new_vector)
        logger.newline()
        logger.debug(f"Updated vector ID:")
        logger.success(f"{vector_id}")
        save_file(vector_id, f"{OUTPUT_DIR}/updated_vector_id.txt")

        # Update multiple vectors at once
        updates = {vid: np.random.rand(VECTOR_DIM).tolist()
                   for vid in vector_ids}
        client.update_vector_by_ids(TABLE_NAME, updates)
        logger.newline()
        logger.debug(f"Updated vectors with IDs:")
        logger.success(vector_ids)
        save_file(vector_ids, f"{OUTPUT_DIR}/updated_vector_ids.json")

        # Search for top 3 most similar vectors
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        similar_vectors = client.search_similar(
            TABLE_NAME, query_vector, top_k=3)
        logger.newline()
        logger.debug(f"Top 3 similar vectors:")
        logger.success(similar_vectors)
        save_file(similar_vectors, f"{OUTPUT_DIR}/similar_vectors.json")

        # Log full database summary before cleanup
        db_summary = client.get_database_summary()
        logger.newline()
        logger.debug("Full database summary:")
        logger.success(f"{db_summary}")
        save_file(db_summary, f"{OUTPUT_DIR}/db_summary.json")

        # Cleanup: Drop all rows in the table
        client.drop_all_rows(TABLE_NAME)
        logger.newline()
        logger.warning("Deleted all rows.")

        # Cleanup: Drop all tables in the database
        client.delete_all_tables()
        logger.newline()
        logger.warning("Deleted all tables.")

        # Test deleting the entire database
        client.delete_db(confirm=True)
        logger.newline()
        logger.warning("Deleted the entire database.")

    except Exception as e:
        logger.newline()
        logger.error(f"Transaction failed:\n{e}")
