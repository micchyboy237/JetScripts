from jet.logger import logger
from langchain_community.chat_models import ChatZhipuAI
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
# Zhipu AI

>[Zhipu AI](https://www.zhipuai.cn/en/aboutus), originating from the technological
> advancements of `Tsinghua University's Computer Science Department`,
> is an artificial intelligence company with the mission of teaching machines
> to think like humans. Its world-leading AI team has developed the cutting-edge
> large language and multimodal models and built the high-precision billion-scale
> knowledge graphs, the combination of which uniquely empowers us to create a powerful
> data- and knowledge-driven cognitive engine towards artificial general intelligence.


## Chat models

See a [usage example](/docs/integrations/chat/zhipuai).
"""
logger.info("# Zhipu AI")


logger.info("\n\n[DONE]", bright=True)