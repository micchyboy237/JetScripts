from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import E2BDataAnalysisTool
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
# E2B Data Analysis

[E2B's cloud environments](https://e2b.dev) are great runtime sandboxes for LLMs.

E2B's Data Analysis sandbox allows for safe code execution in a sandboxed environment. This is ideal for building tools such as code interpreters, or Advanced Data Analysis like in ChatGPT.

E2B Data Analysis sandbox allows you to:
- Run Python code
- Generate charts via matplotlib
- Install Python packages dynamically during runtime
- Install system packages dynamically during runtime
- Run shell commands
- Upload and download files

We'll create a simple Ollama agent that will use E2B's Data Analysis sandbox to perform analysis on a uploaded files using Python.

Get your Ollama API key and [E2B API key here](https://e2b.dev/docs/getting-started/api-key) and set them as environment variables.

You can find the full API documentation [here](https://e2b.dev/docs).

You'll need to install `e2b` to get started:
"""
logger.info("# E2B Data Analysis")

# %pip install --upgrade --quiet  langchain e2b langchain-community




os.environ["E2B_API_KEY"] = "<E2B_API_KEY>"
# os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"

"""
When creating an instance of the `E2BDataAnalysisTool`, you can pass callbacks to listen to the output of the sandbox. This is useful, for example, when creating more responsive UI. Especially with the combination of streaming output from LLMs.
"""
logger.info("When creating an instance of the `E2BDataAnalysisTool`, you can pass callbacks to listen to the output of the sandbox. This is useful, for example, when creating more responsive UI. Especially with the combination of streaming output from LLMs.")

def save_artifact(artifact):
    logger.debug("New matplotlib chart generated:", artifact.name)
    file = artifact.download()
    basename = os.path.basename(artifact.name)

    with open(f"./charts/{basename}", "wb") as f:
        f.write(file)


e2b_data_analysis_tool = E2BDataAnalysisTool(
    env_vars={"MY_SECRET": "secret_value"},
    on_stdout=lambda stdout: logger.debug("stdout:", stdout),
    on_stderr=lambda stderr: logger.debug("stderr:", stderr),
    on_artifact=save_artifact,
)

"""
Upload an example CSV data file to the sandbox so we can analyze it with our agent. You can use for example [this file](https://storage.googleapis.com/e2b-examples/netflix.csv) about Netflix tv shows.
"""
logger.info("Upload an example CSV data file to the sandbox so we can analyze it with our agent. You can use for example [this file](https://storage.googleapis.com/e2b-examples/netflix.csv) about Netflix tv shows.")

with open("./netflix.csv") as f:
    remote_path = e2b_data_analysis_tool.upload_file(
        file=f,
        description="Data about Netflix tv shows including their title, category, director, release date, casting, age rating, etc.",
    )
    logger.debug(remote_path)

"""
Create a `Tool` object and initialize the Langchain agent.
"""
logger.info("Create a `Tool` object and initialize the Langchain agent.")

tools = [e2b_data_analysis_tool.as_tool()]

llm = ChatOllama(model="llama3.2")
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

"""
Now we can ask the agent questions about the CSV file we uploaded earlier.
"""
logger.info("Now we can ask the agent questions about the CSV file we uploaded earlier.")

agent.run(
    "What are the 5 longest movies on netflix released between 2000 and 2010? Create a chart with their lengths."
)

"""
E2B also allows you to install both Python and system (via `apt`) packages dynamically during runtime like this:
"""
logger.info("E2B also allows you to install both Python and system (via `apt`) packages dynamically during runtime like this:")

e2b_data_analysis_tool.install_python_packages("pandas")

"""
Additionally, you can download any file from the sandbox like this:
"""
logger.info("Additionally, you can download any file from the sandbox like this:")

files_in_bytes = e2b_data_analysis_tool.download_file("/home/user/netflix.csv")

"""
Lastly, you can run any shell command inside the sandbox via `run_command`.
"""
logger.info("Lastly, you can run any shell command inside the sandbox via `run_command`.")

e2b_data_analysis_tool.run_command("sudo apt update")
e2b_data_analysis_tool.install_system_packages("sqlite3")

output = e2b_data_analysis_tool.run_command("sqlite3 --version")
logger.debug("version: ", output["stdout"])
logger.debug("error: ", output["stderr"])
logger.debug("exit code: ", output["exit_code"])

"""
When your agent is finished, don't forget to close the sandbox
"""
logger.info("When your agent is finished, don't forget to close the sandbox")

e2b_data_analysis_tool.close()

logger.info("\n\n[DONE]", bright=True)