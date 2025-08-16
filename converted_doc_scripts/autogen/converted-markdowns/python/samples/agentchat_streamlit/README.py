

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Streamlit AgentChat Sample Application

This is a sample AI chat assistant built with [Streamlit](https://streamlit.io/)

## Setup

Install the `streamlit` package with the following command:
"""
logger.info("# Streamlit AgentChat Sample Application")

pip install streamlit

"""
To use Azure Ollama models or models hosted on Ollama-compatible API endpoints,
you need to install the `autogen-ext[openai,azure]` package. You can install it with the following command:
"""
logger.info("To use Azure Ollama models or models hosted on Ollama-compatible API endpoints,")

pip install "autogen-ext[openai,azure]"

"""
Create a new file named `model_config.yml` in the the same directory as the script
to configure the model you want to use.

For example, to use `llama3.1` model from Azure Ollama, you can use the following configuration:
"""
logger.info("Create a new file named `model_config.yml` in the the same directory as the script")

provider: autogen_ext.models.openai.AzureOllamaChatCompletionClient
config:
  azure_deployment: "llama3.1"
  model: llama3.1
  api_version: REPLACE_WITH_MODEL_API_VERSION
  azure_endpoint: REPLACE_WITH_MODEL_ENDPOINT
  api_key: REPLACE_WITH_MODEL_API_KEY

"""
For more information on how to configure the model and use other providers,
please refer to the [Models documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html).

## Run

Run the following command to start the web application:
"""
logger.info("## Run")

streamlit run main.py

logger.info("\n\n[DONE]", bright=True)