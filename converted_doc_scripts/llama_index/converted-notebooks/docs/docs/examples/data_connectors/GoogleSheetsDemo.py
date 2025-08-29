from jet.logger import CustomLogger
from llama_index.readers.google import GoogleSheetsReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Google Sheets Reader**
Demonstrates Google Sheets Reader in LlamaIndex


*   Make Sure you have token.json or credentials.json file in the Environment, More on that [here](https://developers.google.com/workspace/guides/create-credentials)
"""
logger.info("# **Google Sheets Reader**")


"""
Load Sheets as a List of Pandas Dataframe
"""
logger.info("Load Sheets as a List of Pandas Dataframe")

list_of_sheets = ["1ZF5iIeLLqROHbHsb1vOeRaLWKIgLU7rDDTSOZaqjpk0"]
sheets = GoogleSheetsReader()
dataframes = sheets.load_data_in_pandas(list_of_sheets)

dataframes[0]

"""
Or Load Sheets as a List of Document Objects
"""
logger.info("Or Load Sheets as a List of Document Objects")

documents = sheets.load_data(list_of_sheets)

logger.info("\n\n[DONE]", bright=True)