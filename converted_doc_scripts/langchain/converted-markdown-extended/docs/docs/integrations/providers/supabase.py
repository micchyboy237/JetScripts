from jet.logger import logger
from langchain_community.vectorstores import SupabaseVectorStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Supabase (Postgres)

>[Supabase](https://supabase.com/docs) is an open-source `Firebase` alternative.
> `Supabase` is built on top of `PostgreSQL`, which offers strong `SQL`
> querying capabilities and enables a simple interface with already-existing tools and frameworks.

>[PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL) also known as `Postgres`,
> is a free and open-source relational database management system (RDBMS)
> emphasizing extensibility and `SQL` compliance.

## Installation and Setup

We need to install `supabase` python package.
"""
logger.info("# Supabase (Postgres)")

pip install supabase

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/supabase).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)