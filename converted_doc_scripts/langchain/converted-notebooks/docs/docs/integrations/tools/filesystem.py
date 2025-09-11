from jet.logger import logger
from langchain_community.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory
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
# File System

LangChain provides tools for interacting with a local file system out of the box. This notebook walks through some of them.

**Note:** these tools are not recommended for use outside a sandboxed environment!
"""
logger.info("# File System")

# %pip install -qU langchain-community

"""
First, we'll import the tools.
"""
logger.info("First, we'll import the tools.")



working_directory = TemporaryDirectory()

"""
## The FileManagementToolkit

If you want to provide all the file tooling to your agent, it's easy to do so with the toolkit. We'll pass the temporary directory in as a root directory as a workspace for the LLM.

It's recommended to always pass in a root directory, since without one, it's easy for the LLM to pollute the working directory, and without one, there isn't any validation against
straightforward prompt injection.
"""
logger.info("## The FileManagementToolkit")

toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)  # If you don't provide a root_dir, operations will default to the current working directory
toolkit.get_tools()

"""
### Selecting File System Tools

If you only want to select certain tools, you can pass them in as arguments when initializing the toolkit, or you can individually initialize the desired tools.
"""
logger.info("### Selecting File System Tools")

tools = FileManagementToolkit(
    root_dir=str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
tools

read_tool, write_tool, list_tool = tools
write_tool.invoke({"file_path": "example.txt", "text": "Hello World!"})

list_tool.invoke({})

logger.info("\n\n[DONE]", bright=True)