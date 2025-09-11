from jet.logger import logger
from smolagents import CodeAgent, LiteLLMRouterModel, WebSearchTool
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




# Make sure to setup the necessary environment variables!

llm_loadbalancer_model_list = [
    {
        "model_name": "model-group-1",
        "litellm_params": {
            "model": "llama3.2",
#             "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "model-group-1",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region_name": os.getenv("AWS_REGION"),
        },
    },
    # {
    #     "model_name": "model-group-2",
    #     "litellm_params": {
    #         "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    #         "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    #         "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    #         "aws_region_name": os.getenv("AWS_REGION"),
    #     },
    # },
]


model = LiteLLMRouterModel(
    model_id="model-group-1",
    model_list=llm_loadbalancer_model_list,
    client_kwargs={"routing_strategy": "simple-shuffle"},
)
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True, return_full_result=True)

full_result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

logger.debug(full_result)

logger.info("\n\n[DONE]", bright=True)