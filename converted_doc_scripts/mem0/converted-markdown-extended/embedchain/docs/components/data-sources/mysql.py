from embedchain.chunkers.mysql import MySQLChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.loaders.mysql import MySQLLoader
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
title: '🐬 MySQL'
---

1. Setup the MySQL loader by configuring the SQL db.
"""
logger.info("title: '🐬 MySQL'")


config = {
    "host": "host",
    "port": "port",
    "database": "database",
    "user": "username",
    "password": "password",
}

mysql_loader = MySQLLoader(config=config)

"""
For more details on how to setup with valid config, check MySQL [documentation](https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html).

2. Once you setup the loader, you can create an app and load data using the above MySQL loader
"""
logger.info("For more details on how to setup with valid config, check MySQL [documentation](https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html).")


app = App()

app.add("SELECT * FROM table_name;", data_type='mysql', loader=mysql_loader)

response = app.query(question)

"""
NOTE: The `add` function of the app will accept any executable query to load data. DO NOT pass the `CREATE`, `INSERT` queries in `add` function.

3. We automatically create a chunker to chunk your SQL data, however if you wish to provide your own chunker class. Here is how you can do that:
"""
logger.info("NOTE: The `add` function of the app will accept any executable query to load data. DO NOT pass the `CREATE`, `INSERT` queries in `add` function.")


mysql_chunker_config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
mysql_chunker = MySQLChunker(config=mysql_chunker_config)

app.add("SELECT * FROM table_name;", data_type='mysql', loader=mysql_loader, chunker=mysql_chunker)

logger.info("\n\n[DONE]", bright=True)