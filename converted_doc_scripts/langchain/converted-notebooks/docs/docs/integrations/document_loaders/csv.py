from jet.logger import logger
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
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
# CSV

>A [comma-separated values (CSV)](https://en.wikipedia.org/wiki/Comma-separated_values) file is a delimited text file that uses a comma to separate values. Each line of the file is a data record. Each record consists of one or more fields, separated by commas.

Load [csv](https://en.wikipedia.org/wiki/Comma-separated_values) data with a single row per document.
"""
logger.info("# CSV")


loader = CSVLoader(file_path="./example_data/mlb_teams_2012.csv")

data = loader.load()

logger.debug(data)

"""
## Customizing the csv parsing and loading

See the [csv module](https://docs.python.org/3/library/csv.html) documentation for more information of what csv args are supported.
"""
logger.info("## Customizing the csv parsing and loading")

loader = CSVLoader(
    file_path="./example_data/mlb_teams_2012.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["MLB Team", "Payroll in millions", "Wins"],
    },
)

data = loader.load()

logger.debug(data)

"""
## Specify a column to identify the document source

Use the `source_column` argument to specify a source for the document created from each row. Otherwise `file_path` will be used as the source for all documents created from the CSV file.

This is useful when using documents loaded from CSV files for chains that answer questions using sources.
"""
logger.info("## Specify a column to identify the document source")

loader = CSVLoader(file_path="./example_data/mlb_teams_2012.csv", source_column="Team")

data = loader.load()

logger.debug(data)

"""
## `UnstructuredCSVLoader`

You can also load the table using the `UnstructuredCSVLoader`. One advantage of using `UnstructuredCSVLoader` is that if you use it in `"elements"` mode, an HTML representation of the table will be available in the metadata.
"""
logger.info("## `UnstructuredCSVLoader`")


loader = UnstructuredCSVLoader(
    file_path="example_data/mlb_teams_2012.csv", mode="elements"
)
docs = loader.load()

logger.debug(docs[0].metadata["text_as_html"])

logger.info("\n\n[DONE]", bright=True)