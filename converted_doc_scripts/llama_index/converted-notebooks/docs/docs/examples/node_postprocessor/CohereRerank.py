from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.postprocessor.cohere_rerank import CohereRerank
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/CohereRerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Cohere Rerank

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Cohere Rerank")

# %pip install llama-index > /dev/null
# %pip install llama-index-postprocessor-cohere-rerank > /dev/null


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents=documents)

"""
#### Retrieve top 10 most relevant nodes, then filter with Cohere Rerank
"""
logger.info("#### Retrieve top 10 most relevant nodes, then filter with Cohere Rerank")



api_key = os.environ["COHERE_API_KEY"]
cohere_rerank = CohereRerank(api_key=api_key, top_n=2)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

pprint_response(response, show_source=True)

"""
### Directly retrieve top 2 most similar nodes
"""
logger.info("### Directly retrieve top 2 most similar nodes")

query_engine = index.as_query_engine(
    similarity_top_k=2,
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

"""
Retrieved context is irrelevant and response is hallucinated.
"""
logger.info("Retrieved context is irrelevant and response is hallucinated.")

pprint_response(response, show_source=True)

logger.info("\n\n[DONE]", bright=True)