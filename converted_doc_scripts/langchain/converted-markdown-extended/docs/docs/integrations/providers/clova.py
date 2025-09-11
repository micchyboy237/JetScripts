from jet.logger import logger
from langchain_community.embeddings import ClovaEmbeddings
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
# Clova

>[CLOVA Studio](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary) is a service
> of [Naver Cloud Platform](https://www.ncloud.com/) that uses `HyperCLOVA` language models,
> a hyperscale AI technology, to output phrases generated through AI technology based on user input.


## Embedding models

See [installation instructions and usage example](/docs/integrations/text_embedding/clova).
"""
logger.info("# Clova")


logger.info("\n\n[DONE]", bright=True)