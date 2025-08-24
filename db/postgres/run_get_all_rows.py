import os
import shutil
from jet.db.postgres.client import PostgresClient
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.collection_utils import group_by

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# DBNAME = "jobs_db1"
# TABLE_NAME = "embeddings_data"
DBNAME = "mlx_agents_chat_history_db1"
TABLE_NAME = "messages"

with PostgresClient(
    dbname=DBNAME,
) as client:
    all_rows = client.get_rows(TABLE_NAME)
    logger.newline()
    logger.debug(f"All rows in {TABLE_NAME}:")
    logger.success(format_json(all_rows))
    save_file(
        {
            "table": TABLE_NAME,
            "count": len(all_rows),
            "rows": all_rows  # Directly save all columns except embedding
        },
        f"{OUTPUT_DIR}/all_rows.json"
    )

    grouped_rows = group_by(all_rows, "session_id")
    save_file(
        {
            "table": TABLE_NAME,
            "group_key": "session_id",
            "count": len(grouped_rows),
            "rows": grouped_rows  # Directly save all columns except embedding
        },
        f"{OUTPUT_DIR}/grouped_rows.json"
    )
