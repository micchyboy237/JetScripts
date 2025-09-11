from IPython.display import display
from PIL import Image
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
import base64
import io
import json
import matplotlib.pyplot as plt
import numpy as np
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
# Azure Container Apps dynamic sessions

Azure Container Apps dynamic sessions provides a secure and scalable way to run a Python code interpreter in Hyper-V isolated sandboxes. This allows your agents to run potentially untrusted code in a secure environment. The code interpreter environment includes many popular Python packages, such as NumPy, pandas, and scikit-learn. See the [Azure Container App docs](https://learn.microsoft.com/en-us/azure/container-apps/sessions-code-interpreter) for more info on how sessions work.

## Setup

By default, the `SessionsPythonREPLTool` tool uses `DefaultAzureCredential` to authenticate with Azure. Locally, it'll use your credentials from the Azure CLI or VS Code. Install the Azure CLI and log in with `az login` to authenticate.

To use the code interpreter you'll also need to create a session pool, which you can do by following the instructions [here](https://learn.microsoft.com/en-us/azure/container-apps/sessions-code-interpreter?tabs=azure-cli#create-a-session-pool-with-azure-cli). Once that's done you should have a pool management session endpoint, which you'll need to set below:
"""
logger.info("# Azure Container Apps dynamic sessions")

# import getpass

# POOL_MANAGEMENT_ENDPOINT = getpass.getpass()

"""
You'll also need to install the `langchain-azure-dynamic-sessions` package:
"""
logger.info("You'll also need to install the `langchain-azure-dynamic-sessions` package:")

# %pip install -qU langchain-azure-dynamic-sessions langchain-ollama langchainhub langchain langchain-community

"""
## Use tool

Instantiate and use tool:
"""
logger.info("## Use tool")


tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
tool.invoke("6 * 7")

"""
Invoking the tool will return a json string with the result of the code, along with any stdout and stderr outputs. To get the raw dictionary results, use the `execute()` method:
"""
logger.info("Invoking the tool will return a json string with the result of the code, along with any stdout and stderr outputs. To get the raw dictionary results, use the `execute()` method:")

tool.execute("6 * 7")

"""
## Upload data

If we want to perform computation over specific data, we can use the `upload_file()` functionality to upload data to our session. You can upload data either via the `data: BinaryIO` arg or via the `local_file_path: str` arg (which points to a local file on your system). The data is automatically uploaded to the "/mnt/data/" directory in the sessions container. You can get the full file path via the upload metadata returned by `upload_file()`.
"""
logger.info("## Upload data")


data = {"important_data": [1, 10, -1541]}
binary_io = io.BytesIO(json.dumps(data).encode("ascii"))

upload_metadata = tool.upload_file(
    data=binary_io, remote_file_path="important_data.json"
)

code = f"""

with open("{upload_metadata.full_path}") as f:
    data = json.load(f)

sum(data['important_data'])
"""
tool.execute(code)

"""
## Handling image results

Dynamic sessions results can include image outputs as base64 encoded strings. In these cases the value of 'result' will be a dictionary with keys "type" (which will be "image"), "format (the format of the image), and "base64_data".
"""
logger.info("## Handling image results")

code = """

x = np.linspace(-1, 1, 400)

y = np.sin(x)

plt.plot(x, y)

plt.title('Plot of sin(x) from -1 to 1')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.grid(True)
plt.show()
"""

result = tool.execute(code)
result["result"].keys()

result["result"]["type"], result["result"]["format"]

"""
We can decode the image data and display it:
"""
logger.info("We can decode the image data and display it:")



base64_str = result["result"]["base64_data"]
img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
display(img)

"""
## Simple agent example
"""
logger.info("## Simple agent example")


llm = ChatOllama(model="llama3.2")
prompt = hub.pull("hwchase17/ollama-functions-agent")
agent = create_tool_calling_agent(llm, [tool], prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=[tool], verbose=True, handle_parsing_errors=True
)

response = agent_executor.invoke(
    {
        "input": "what's sin of pi . if it's negative generate a random number between 0 and 5. if it's positive between 5 and 10."
    }
)

"""
## LangGraph data analyst agent

For a more complex agent example check out the LangGraph data analyst example https://github.com/langchain-ai/langchain/blob/master/cookbook/azure_container_apps_dynamic_sessions_data_analyst.ipynb
"""
logger.info("## LangGraph data analyst agent")


logger.info("\n\n[DONE]", bright=True)