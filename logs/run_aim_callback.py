import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SummaryIndex
from llama_index.callbacks.aim import AimCallback
from llama_index.core.callbacks import CallbackManager
from jet.logger import logger
from jet.vectors import SettingsManager


# %pip install llama-index-callbacks-aim

# !pip install llama-index


# !mkdir - p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' - O 'data/paul_graham/paul_graham_essay.txt'

file_name = os.path.splitext(os.path.basename(__file__))[0]
repo_dir = f"./generated/{file_name}"

if __name__ == "__main__":
    # Settings initialization
    settings_manager = SettingsManager.create()

    data_path = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/data/paul_graham"
    docs = SimpleDirectoryReader(data_path).load_data()

    aim_callback = AimCallback(repo=repo_dir)
    callback_manager = CallbackManager([aim_callback])

    index = SummaryIndex.from_documents(
        docs, callback_manager=callback_manager, show_progress=True)
    logger.debug(index.summary)

    query_engine = index.as_query_engine(llm=settings_manager.llm)

    response = query_engine.query("What did the author do growing up?")
    logger.success(response)
