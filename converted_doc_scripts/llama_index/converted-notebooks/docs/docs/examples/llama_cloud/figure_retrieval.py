from IPython.display import Image, display
from jet.logger import CustomLogger
from llama_cloud.types import LlamaParseParameters
from llama_index.core.schema import ImageNode
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
import base64
import os
import shutil
import tempfile


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LlamaCloud Page Figure Retrieval
This notebook shows an example of retrieving images embedded within a sample PDF document.
More docs on using this feature can be found on the [LlamaCloud docs page](https://docs.cloud.llamaindex.ai/llamacloud/retrieval/images).
"""
logger.info("# LlamaCloud Page Figure Retrieval")

# %pip install llama-index llama-index-llms-ollama llama-cloud llama-index-indices-managed-llama-cloud

"""
### Create an Index and upload the figures PDF to it
"""
logger.info("### Create an Index and upload the figures PDF to it")


api_key = os.environ["LLAMA_CLOUD_API_KEY"]
org_id = os.environ.get("LLAMA_CLOUD_ORGANIZATION_ID")
# openai_api_key = os.environ["OPENAI_API_KEY"]


logger.debug(os.getcwd())


embedding_config = {
    "type": "OPENAI_EMBEDDING",
    "component": {
        "api_key": openai_api_key,
        "model_name": "text-embedding-ada-002",  # You can choose any OllamaFunctionCalling Embedding model
    },
}

index = LlamaCloudIndex.create_index(
    name="my_index",
    organization_id=org_id,
    api_key=api_key,
    embedding_config=embedding_config,
    llama_parse_parameters=LlamaParseParameters(
        take_screenshot=True,
        extract_layout=True,
    ),
)


image_figure_slides_path = "../data/figures/image_figure_slides.pdf"
index.upload_file(
    image_figure_slides_path, wait_for_ingestion=True, raise_on_error=True
)

"""
### Start Retrieving Page Figures
"""
logger.info("### Start Retrieving Page Figures")


retriever = index.as_retriever(
    retrieve_page_figure_nodes=True, dense_similarity_top_k=1
)

nodes = retriever.retrieve("Sample query")

image_nodes = [n.node for n in nodes if isinstance(n.node, ImageNode)]

for img_node in image_nodes:
    logger.debug(img_node.metadata)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file.write(base64.b64decode(img_node.image))
        logger.debug(f"Image saved to {temp_file.name}")
        display(Image(filename=temp_file.name))

logger.info("\n\n[DONE]", bright=True)