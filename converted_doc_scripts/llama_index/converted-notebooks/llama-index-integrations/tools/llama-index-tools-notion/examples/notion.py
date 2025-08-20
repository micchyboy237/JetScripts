from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.notion.base import NotionToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"



notion_token = "secret_your-key"
tool_spec = NotionToolSpec(integration_token=notion_token)

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)


ctx = Context(agent)

logger.debug(
    await agent.run(
        "append the heading 'I Am Legend' to the movies page",
        ctx=ctx,
    )
)

logger.debug(
    await agent.run(
        "append the heading 'I Am Legend' to the movies page",
        ctx=ctx,
    )
)

logger.info("\n\n[DONE]", bright=True)