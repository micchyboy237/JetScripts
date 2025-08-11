import os
import shutil
import json
from jet.db.postgres.client import PostgresClient
from jet.file.utils import save_file
from jet.logger import logger

TABLE_NAME = "data_rows"
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

with PostgresClient(
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

        client.create_table(TABLE_NAME)
        table_metadata = client.get_table_metadata(TABLE_NAME)
        logger.newline()
        logger.debug(f"Metadata for table {TABLE_NAME}:")
        logger.success(f"{table_metadata}")
        save_file(table_metadata,
                  f"{OUTPUT_DIR}/table_metadata_after_creation.json")

        # Example: Create a single row with metadata and nested dict
        single_row_data = {
            "metadata": "example metadata",
            "score": 95.5,
            "is_active": True,
            "details": {"key1": "value1", "key2": {"nested_key": 42}},
            "custom_field": "test_value"
        }
        single_row = client.create_row(TABLE_NAME, single_row_data)
        logger.newline()
        logger.debug(f"Created single row:")
        logger.success(f"{single_row}")
        save_file(single_row, f"{OUTPUT_DIR}/created_single_row.json")

        # Example: Create multiple rows with additional columns and nested dicts
        multiple_rows_data = [
            {
                "metadata": f"row_{i}",
                "score": 90.0 + i,
                "is_active": i % 2 == 0,
                "details": {"index": i, "data": {"value": f"test_{i}"}},
                "custom_field": f"custom_{i}"
            } for i in range(3)
        ]
        multiple_rows = client.create_rows(TABLE_NAME, multiple_rows_data)
        logger.newline()
        logger.debug(f"Created multiple rows:")
        logger.success(f"{multiple_rows}")
        save_file(
            multiple_rows,
            f"{OUTPUT_DIR}/created_multiple_rows.json"
        )

        # Example: Update a single row with new values
        updated_single_row_data = {
            "id": single_row["id"],
            "metadata": "updated metadata",
            "score": 98.0,
            "is_active": False,
            "details": {"key1": "updated_value1", "key2": {"nested_key": 100}},
            "custom_field": "updated_test_value"
        }
        updated_single_row = client.update_row(
            TABLE_NAME, single_row["id"], updated_single_row_data)
        logger.newline()
        logger.debug(f"Updated single row for ID {single_row['id']}:")
        logger.success(f"{updated_single_row}")
        save_file(updated_single_row, f"{OUTPUT_DIR}/updated_single_row.json")

        # Example: Retrieve single row by ID
        retrieved_row = client.get_row(TABLE_NAME, single_row["id"])
        logger.newline()
        logger.debug(f"Retrieved single row for ID {single_row['id']}:")
        logger.success(f"{retrieved_row}")
        save_file(retrieved_row, f"{OUTPUT_DIR}/retrieved_single_row.json")

        # Example: Update multiple rows with new values
        updated_multiple_rows_data = [
            {
                "id": row["id"],
                "metadata": f"updated_row_{i}",
                "score": 95.0 + i,
                "is_active": (i % 2 != 0),
                "details": {"index": i, "data": {"value": f"updated_test_{i}"}},
                "custom_field": f"updated_custom_{i}"
            } for i, row in enumerate(multiple_rows)
        ]
        updated_multiple_rows = client.update_rows(
            TABLE_NAME, updated_multiple_rows_data)
        logger.newline()
        logger.debug(f"Updated multiple rows:")
        logger.success(f"{updated_multiple_rows}")
        save_file(
            updated_multiple_rows,
            f"{OUTPUT_DIR}/updated_multiple_rows.json"
        )

        # Example: Retrieve all rows
        all_rows = client.get_rows(TABLE_NAME)
        logger.newline()
        logger.debug(f"All rows in {TABLE_NAME}:")
        logger.success(f"{all_rows}")
        save_file(
            {
                "table": TABLE_NAME,
                "count": len(all_rows),
                "rows": all_rows
            },
            f"{OUTPUT_DIR}/all_rows.json"
        )

        # Example: Retrieve filtered rows by IDs
        selected_ids = [single_row["id"],
                        multiple_rows[0]["id"], "non_existent_id"]
        filtered_rows = client.get_rows(TABLE_NAME, ids=selected_ids)
        logger.newline()
        logger.debug(f"Filtered rows for IDs {selected_ids}:")
        logger.success(f"{filtered_rows}")
        save_file(
            {
                "table": TABLE_NAME,
                "count": len(filtered_rows),
                "selected_ids": selected_ids,
                "rows": filtered_rows
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
