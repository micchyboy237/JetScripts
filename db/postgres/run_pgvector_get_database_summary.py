import os
import shutil
from jet.file.utils import save_file
from jet.logger import logger
import numpy as np
from jet.db.postgres.pgvector import PgVectorClient
from shared.job_helpers import DEFAULT_JOBS_DB_NAME

# Database configuration
TABLE_NAME = "embeddings"
VECTOR_DIM = 3

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

with PgVectorClient(
    dbname=DEFAULT_JOBS_DB_NAME,
    overwrite_db=False,
) as client:
    try:
        # Log full database summary before cleanup
        db_summary = client.get_database_summary()
        logger.newline()
        logger.debug("Full database summary:")
        logger.success(f"{db_summary}")
        save_file(db_summary, f"{OUTPUT_DIR}/jobs_db_summary.json")

    except Exception as e:
        logger.newline()
        logger.error(f"Transaction failed:\n{e}")
