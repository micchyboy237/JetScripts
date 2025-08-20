import asyncio
from jet.transformers.formatters import format_json
from box_sdk_gen import DeveloperTokenConfig, BoxDeveloperTokenAuth, BoxClient
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.box import BoxAIExtractToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


BOX_DEV_TOKEN = "your_box_dev_token"

config = DeveloperTokenConfig(BOX_DEV_TOKEN)
auth = BoxDeveloperTokenAuth(config)
box_client = BoxClient(auth)


# os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"




document_id = "test_txt_invoice_id"
ai_prompt = (
    '{"doc_type","date","total","vendor","invoice_number","purchase_order_number"}'
)

box_tool = BoxAIExtractToolSpec(box_client=box_client)

agent = FunctionAgent(
    tools=box_tool.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

async def run_async_code_c0040448():
    async def run_async_code_5f5f4a40():
        answer = await agent.run(f"{ai_prompt} for {document_id}")
        return answer
    answer = asyncio.run(run_async_code_5f5f4a40())
    logger.success(format_json(answer))
    return answer
answer = asyncio.run(run_async_code_c0040448())
logger.success(format_json(answer))
logger.debug(answer)

logger.info("\n\n[DONE]", bright=True)