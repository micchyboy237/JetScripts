from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.postprocessor import (
PIINodePostprocessor,
NERPIINodePostprocessor,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.postprocessor.presidio import PresidioPIINodePostprocessor
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/PII.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PII Masking

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# PII Masking")

# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-huggingface

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


text = """
Hello Paulo Santos. The latest statement for your credit card account \
1111-0000-1111-0000 was mailed to 123 Any Street, Seattle, WA 98109.
"""
node = TextNode(text=text)

"""
### Option 1: Use NER Model for PII Masking

Use a Hugging Face NER model for PII Masking
"""
logger.info("### Option 1: Use NER Model for PII Masking")

processor = NERPIINodePostprocessor()


new_nodes = processor.postprocess_nodes([NodeWithScore(node=node)])

new_nodes[0].node.get_text()

new_nodes[0].node.metadata["__pii_node_info__"]

"""
### Option 2: Use LLM for PII Masking

NOTE: You should be using a *local* LLM model for PII masking. The example shown is using MLX, but normally you'd use an LLM running locally, possibly from huggingface. Examples for local LLMs are [here](https://gpt-index.readthedocs.io/en/latest/how_to/customization/custom_llms.html#example-using-a-huggingface-llm).
"""
logger.info("### Option 2: Use LLM for PII Masking")


processor = PIINodePostprocessor(llm=MLX())


new_nodes = processor.postprocess_nodes([NodeWithScore(node=node)])

new_nodes[0].node.get_text()

new_nodes[0].node.metadata["__pii_node_info__"]

"""
### Option 3: Use Presidio for PII Masking

Use presidio to identify and anonymize PII
"""
logger.info("### Option 3: Use Presidio for PII Masking")

text = """
Hello Paulo Santos. The latest statement for your credit card account \
4095-2609-9393-4932 was mailed to Seattle, WA 98109. \
IBAN GB90YNTU67299444055881 and social security number is 474-49-7577 were verified on the system. \
Further communications will be sent to paulo@presidio.site
"""
presidio_node = TextNode(text=text)


processor = PresidioPIINodePostprocessor()


presidio_new_nodes = processor.postprocess_nodes(
    [NodeWithScore(node=presidio_node)]
)

presidio_new_nodes[0].node.get_text()

presidio_new_nodes[0].node.metadata["__pii_node_info__"]

"""
### Feed Nodes to Index
"""
logger.info("### Feed Nodes to Index")

index = VectorStoreIndex([n.node for n in new_nodes])

response = index.as_query_engine().query(
    "What address was the statement mailed to?"
)
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)