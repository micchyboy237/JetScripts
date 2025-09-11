from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_googledrive.tools.google_drive.tool import GoogleDriveSearchTool
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper
from langgraph.prebuilt import create_react_agent
import logging
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
# Google Drive

This notebook walks through connecting a LangChain to the `Google Drive API`.

## Prerequisites

1. Create a Google Cloud project or use an existing project
1. Enable the [Google Drive API](https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com)
1. [Authorize credentials for desktop app](https://developers.google.com/drive/api/quickstart/python#authorize_credentials_for_a_desktop_application)
1. `pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib`

## Instructions for retrieving your Google Docs data
By default, the `GoogleDriveTools` and `GoogleDriveWrapper` expects the `credentials.json` file to be `~/.credentials/credentials.json`, but this is configurable by setting the `GOOGLE_ACCOUNT_FILE` environment variable to your `custom/path/to/credentials.json`. 
The location of `token.json` use the same directory (or use the parameter `token_path`). Note that `token.json` will be created automatically the first time you use the tool.

`GoogleDriveSearchTool` can retrieve a selection of files with some requests. 

By default, If you use a `folder_id`, all the files inside this folder can be retrieved to `Document`, if the name match the query.
"""
logger.info("# Google Drive")

# %pip install --upgrade --quiet  google-api-python-client google-auth-httplib2 google-auth-oauthlib langchain-community

"""
You can obtain your folder and document id from the URL:

* Folder: https://drive.google.com/drive/u/0/folders/1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5 -> folder id is `"1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5"`
* Document: https://docs.google.com/document/d/1bfaMQ18_i56204VaQDVeAFpqEijJTgvurupdEDiaUQw/edit -> document id is `"1bfaMQ18_i56204VaQDVeAFpqEijJTgvurupdEDiaUQw"`

The special value `root` is for your personal home.
"""
logger.info("You can obtain your folder and document id from the URL:")

folder_id = "root"

"""
By default, all files with these mime-type can be converted to `Document`.
- text/text
- text/plain
- text/html
- text/csv
- text/markdown
- image/png
- image/jpeg
- application/epub+zip
- application/pdf
- application/rtf
- application/vnd.google-apps.document (GDoc)
- application/vnd.google-apps.presentation (GSlide)
- application/vnd.google-apps.spreadsheet (GSheet)
- application/vnd.google.colaboratory (Notebook colab)
- application/vnd.openxmlformats-officedocument.presentationml.presentation (PPTX)
- application/vnd.openxmlformats-officedocument.wordprocessingml.document (DOCX)

It's possible to update or customize this. See the documentation of `GoogleDriveAPIWrapper`.

But, the corresponding packages must installed.
"""
logger.info("By default, all files with these mime-type can be converted to `Document`.")

# %pip install --upgrade --quiet  unstructured langchain-googledrive



os.environ["GOOGLE_ACCOUNT_FILE"] = "custom/path/to/credentials.json"

tool = GoogleDriveSearchTool(
    api_wrapper=GoogleDriveAPIWrapper(
        folder_id=folder_id,
        num_results=2,
        template="gdrive-query-in-folder",  # Search in the body of documents
    )
)


logging.basicConfig(level=logging.INFO)

tool.run("machine learning")

tool.description

"""
## Use the tool within a ReAct agent

In order to create an agent that uses the Google Jobs tool install Langgraph
"""
logger.info("## Use the tool within a ReAct agent")

# %pip install --upgrade --quiet langgraph langchain-ollama

"""
and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPEN_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.
"""
logger.info("and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPEN_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.")



# os.environ["OPENAI_API_KEY"] = "your-ollama-api-key"


llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0)
agent = create_react_agent(llm, tools=[tool])

events = agent.stream(
    {"messages": [("user", "Search in google drive, who is 'Yann LeCun' ?")]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)