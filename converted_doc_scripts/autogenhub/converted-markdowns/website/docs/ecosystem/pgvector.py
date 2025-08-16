

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# PGVector

[PGVector](https://github.com/pgvector/pgvector) is an open-source vector similarity search for Postgres.

- [PGVector + AutoGen Code Examples](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_RetrieveChat_pgvector.ipynb)
"""
logger.info("# PGVector")

logger.info("\n\n[DONE]", bright=True)