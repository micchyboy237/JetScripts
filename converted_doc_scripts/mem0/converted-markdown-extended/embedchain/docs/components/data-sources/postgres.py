from embedchain.chunkers.postgres import PostgresChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.loaders.postgres import PostgresLoader
from embedchain.pipeline import Pipeline as App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '🐘 Postgres'
---

1. Setup the Postgres loader by configuring the postgres db.
"""
logger.info("title: '🐘 Postgres'")


config = {
    "host": "host_address",
    "port": "port_number",
    "dbname": "database_name",
    "user": "username",
    "password": "password",
}

"""
config = {
    "url": "your_postgres_url"
}
"""

postgres_loader = PostgresLoader(config=config)

"""
You can either setup the loader by passing the postgresql url or by providing the config data.
For more details on how to setup with valid url and config, check postgres [documentation](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING:~:text=34.1.1.%C2%A0Connection%20Strings-,%23,-Several%20libpq%20functions).

NOTE: if you provide the `url` field in config, all other fields will be ignored.

2. Once you setup the loader, you can create an app and load data using the above postgres loader
"""
logger.info("You can either setup the loader by passing the postgresql url or by providing the config data.")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"

app = App()

question = "What is Elon Musk's networth?"
response = app.query(question)

app.add("SELECT * FROM table_name;", data_type='postgres', loader=postgres_loader)

response = app.query(question)

"""
NOTE: The `add` function of the app will accept any executable query to load data. DO NOT pass the `CREATE`, `INSERT` queries in `add` function as they will result in not adding any data, so it is pointless.

3. We automatically create a chunker to chunk your postgres data, however if you wish to provide your own chunker class. Here is how you can do that:
"""
logger.info("NOTE: The `add` function of the app will accept any executable query to load data. DO NOT pass the `CREATE`, `INSERT` queries in `add` function as they will result in not adding any data, so it is pointless.")


postgres_chunker_config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
postgres_chunker = PostgresChunker(config=postgres_chunker_config)

app.add("SELECT * FROM table_name;", data_type='postgres', loader=postgres_loader, chunker=postgres_chunker)

logger.info("\n\n[DONE]", bright=True)