from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureMLX
from llama_index.tools.azure_code_interpreter import (
AzureCodeInterpreterToolSpec,
)
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Azure Code Interpreter Tool Spec

This example walks through configuring and using the Azure Code Interpreter tool spec (powered by Azure Dynamic Sessions).
"""
logger.info("# Azure Code Interpreter Tool Spec")

# %pip install llama-index
# %pip install llama-index-llms-azure
# %pip install llama-index-tools-azure-code-interpreter


api_key = "your-azure-openai-api-key"
azure_endpoint = "your-azure-openai-endpoint"
api_version = "azure-api-version"



azure_code_interpreter_spec = AzureCodeInterpreterToolSpec(
    pool_management_endpoint="your-pool-management-endpoint",
    local_save_path="local-file-path-to-save-intermediate-data",
)

llm = AzureMLX(
    model="gpt-35-turbo",
    deployment_name="gpt-35-deploy",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

agent = ReActAgent.from_tools(
    azure_code_interpreter_spec.to_tool_list(), llm=llm, verbose=True
)

logger.debug(azure_code_interpreter_spec.code_interpreter("1+1"))

logger.debug(agent.chat("Tell me the current time in Seattle."))

res = azure_code_interpreter_spec.upload_file(
    local_file_path="./TemperatureData.csv"
)
if len(res) != 0:
    logger.debug(
        agent.chat("Find the highest temperature in the file that I uploaded.")
    )

logger.debug(
    agent.chat(
        "Use the temperature data that I uploaded, create a temperature curve."
    )
)

logger.debug(
    agent.chat(
        "Rearrange the temperature data in a descending order and save it back to the original csv file."
    )
)

azure_code_interpreter_spec.download_file_to_local(
    remote_file_path="TemperatureData.csv",
    local_file_path="/.../SortedTemperatureData.csv",
)

with open("/.../TemperatureData.csv", "r") as f:
    for i in range(10):
        logger.debug(f.readline().strip())

with open("/.../SortedTemperatureData.csv", "r") as f:
    for i in range(10):
        logger.debug(f.readline().strip())

logger.info("\n\n[DONE]", bright=True)