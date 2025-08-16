from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# PGVector

[PGVector](https://github.com/pgvector/pgvector) is an open-source vector similarity search for Postgres.

- [PGVector + AutoGen Code Examples](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_RetrieveChat_pgvector.ipynb)
"""
logger.info("# PGVector")

logger.info("\n\n[DONE]", bright=True)