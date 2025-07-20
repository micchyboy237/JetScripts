import os
import shutil
import json
from jet.file.utils import save_file
from jet.logger import logger
import numpy as np
from jet.db.postgres.pgvector import PgVectorClient

TABLE_NAME = "embeddings"
EMBEDDING_DIM = 3
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

with PgVectorClient(
    dbname="test_db1",
    overwrite_db=True,
) as client:
    try:
        db_metadata = client.get_database_metadata()
        logger.newline()
        logger.debug("Database metadata:")
        logger.success(f"{db_metadata}")
        save_file(db_metadata, f"{OUTPUT_DIR}/db_metadata.json")

        client.delete_all_tables()
        tables = client.get_all_tables()
        logger.newline()
        logger.debug("All tables after deletion:")
        logger.success(f"{tables}")
        save_file(tables, f"{OUTPUT_DIR}/tables_after_deletion.json")

        client.create_table(TABLE_NAME, EMBEDDING_DIM)
        table_metadata = client.get_table_metadata(TABLE_NAME)
        logger.newline()
        logger.debug(f"Metadata for table {TABLE_NAME}:")
        logger.success(f"{table_metadata}")
        save_file(table_metadata,
                  f"{OUTPUT_DIR}/table_metadata_after_creation.json")

        embedding = np.random.rand(EMBEDDING_DIM).tolist()
        embedding_id = client.insert_embedding(TABLE_NAME, embedding)
        logger.newline()
        logger.debug(f"Inserted embedding ID:")
        logger.success(f"{embedding_id}")
        save_file(embedding_id, f"{OUTPUT_DIR}/inserted_embedding_id.txt")

        retrieved_embedding = client.get_embedding_by_id(
            TABLE_NAME, embedding_id)
        logger.newline()
        logger.debug(f"Retrieved embedding:")
        logger.success(f"{retrieved_embedding}")
        save_file(retrieved_embedding.tolist() if retrieved_embedding is not None else None,
                  f"{OUTPUT_DIR}/retrieved_embedding.json")

        embeddings = [np.random.rand(EMBEDDING_DIM).tolist() for _ in range(3)]
        embedding_ids = client.insert_embeddings(TABLE_NAME, embeddings)
        logger.newline()
        logger.debug(f"Inserted multiple embedding IDs:")
        logger.success(f"{embedding_ids}")
        save_file(embedding_ids, f"{OUTPUT_DIR}/inserted_embedding_ids.json")

        all_embeddings = client.get_embeddings(TABLE_NAME)
        logger.newline()
        logger.debug(f"All embeddings in the table:")
        logger.success(f"{all_embeddings}")
        save_file(
            {k: v.tolist() for k, v in all_embeddings.items()},
            f"{OUTPUT_DIR}/all_embeddings.json"
        )

        specific_id = "custom-123"

        selected_ids = [embedding_id, specific_id, "emb-1"]
        filtered_embeddings = client.get_embeddings(
            TABLE_NAME, ids=selected_ids)
        logger.newline()
        logger.debug(f"Filtered embeddings for IDs {selected_ids}:")
        logger.success(f"{filtered_embeddings}")
        save_file(
            {k: v.tolist() for k, v in filtered_embeddings.items()},
            f"{OUTPUT_DIR}/filtered_embeddings.json"
        )

        embedding_count = client.count_embeddings(TABLE_NAME)
        logger.newline()
        logger.debug(f"Total number of embeddings:")
        logger.success(f"{embedding_count}")
        save_file(embedding_count, f"{OUTPUT_DIR}/embedding_count.json")

        table_metadata = client.get_table_metadata(TABLE_NAME)
        logger.newline()
        logger.debug(f"Updated metadata for table {TABLE_NAME}:")
        logger.success(f"{table_metadata}")
        save_file(table_metadata,
                  f"{OUTPUT_DIR}/table_metadata_after_inserts.json")

        specific_embedding = np.random.rand(EMBEDDING_DIM).tolist()
        client.insert_embedding_by_id(
            TABLE_NAME, specific_id, specific_embedding)
        logger.newline()
        logger.debug(f"Inserted embedding with specific ID:")
        logger.success(f"{specific_id}")
        save_file(specific_id, f"{OUTPUT_DIR}/inserted_specific_id.txt")

        retrieved_specific_embedding = client.get_embedding_by_id(
            TABLE_NAME, specific_id)
        logger.newline()
        logger.debug(f"Retrieved embedding for ID {specific_id}:")
        logger.success(f"{retrieved_specific_embedding}")
        save_file(retrieved_specific_embedding.tolist() if retrieved_specific_embedding is not None else None,
                  f"{OUTPUT_DIR}/retrieved_specific_embedding.json")

        specific_embeddings = {
            "emb-1": np.random.rand(EMBEDDING_DIM),
            "emb-2": np.random.rand(EMBEDDING_DIM),
            "emb-3": np.random.rand(EMBEDDING_DIM),
        }
        specific_embeddings_list = {k: v.tolist() if hasattr(
            v, "tolist") else v for k, v in specific_embeddings.items()}
        client.insert_embedding_by_ids(TABLE_NAME, specific_embeddings_list)
        logger.newline()
        logger.debug(f"Inserted multiple embeddings with specific IDs:")
        logger.success(list(specific_embeddings.keys()))
        save_file(list(specific_embeddings.keys()),
                  f"{OUTPUT_DIR}/inserted_specific_ids.json")

        retrieved_specific_embeddings = client.get_embeddings(
            TABLE_NAME, list(specific_embeddings.keys()))
        logger.newline()
        logger.debug(f"Retrieved specific embeddings:")
        logger.success(retrieved_specific_embeddings)
        save_file({k: v.tolist() for k, v in retrieved_specific_embeddings.items()},
                  f"{OUTPUT_DIR}/retrieved_specific_embeddings.json")

        new_embedding = np.random.rand(EMBEDDING_DIM).tolist()
        client.update_embedding_by_id(TABLE_NAME, embedding_id, new_embedding)
        logger.newline()
        logger.debug(f"Updated embedding ID:")
        logger.success(f"{embedding_id}")
        save_file(embedding_id, f"{OUTPUT_DIR}/updated_embedding_id.txt")

        updates = {eid: np.random.rand(EMBEDDING_DIM).tolist()
                   for eid in embedding_ids}
        client.update_embedding_by_ids(TABLE_NAME, updates)
        logger.newline()
        logger.debug(f"Updated embeddings with IDs:")
        logger.success(embedding_ids)
        save_file(embedding_ids, f"{OUTPUT_DIR}/updated_embedding_ids.json")

        query_embedding = np.random.rand(EMBEDDING_DIM).tolist()
        similar_embeddings = client.search_similar(
            TABLE_NAME, query_embedding, top_k=3)
        logger.newline()
        logger.debug(f"Top 3 similar embeddings:")
        logger.success(similar_embeddings)
        save_file(similar_embeddings, f"{OUTPUT_DIR}/similar_embeddings.json")

        all_rows = client.get_rows(TABLE_NAME)
        logger.newline()
        logger.debug(f"All rows in {TABLE_NAME}:")
        logger.success(f"{all_rows}")
        save_file(
            {
                "table": TABLE_NAME,
                "count": len(all_rows),
                "rows": [
                    {"id": row["id"], "embedding": row["embedding"].tolist()}
                    for row in all_rows
                ]
            },
            f"{OUTPUT_DIR}/all_rows.json"
        )

        selected_ids = [embedding_id, specific_id, "emb-1"]
        filtered_rows = client.get_rows(TABLE_NAME, ids=selected_ids)
        logger.newline()
        logger.debug(f"Filtered rows for IDs {selected_ids}:")
        logger.success(f"{filtered_rows}")
        save_file(
            {
                "table": TABLE_NAME,
                "count": len(filtered_rows),
                "selected_ids": selected_ids,
                "rows": [
                    {"id": row["id"], "embedding": row["embedding"].tolist()}
                    for row in filtered_rows
                ]
            },
            f"{OUTPUT_DIR}/filtered_rows.json"
        )

        db_summary = client.get_database_summary()
        logger.newline()
        logger.debug("Full database summary:")
        logger.success(f"{db_summary}")
        save_file(db_summary, f"{OUTPUT_DIR}/db_summary.json")

        client.drop_all_rows(TABLE_NAME)
        logger.newline()
        logger.warning("Deleted all rows.")

        client.delete_all_tables()
        logger.newline()
        logger.warning("Deleted all tables.")

        client.delete_db(confirm=True)
        logger.newline()
        logger.warning("Deleted the entire database.")
    except Exception as e:
        logger.newline()
        logger.error(f"Transaction failed:\n{e}")
