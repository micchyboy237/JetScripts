from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.packs.arize_phoenix_query_engine import ArizePhoenixQueryEnginePack
from llama_index.readers.web import SimpleWebPageReader
from tqdm.auto import tqdm
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
<center>
    <p style="text-align:center">
        <img alt="phoenix logo" src="https://storage.googleapis.com/arize-assets/phoenix/assets/phoenix-logo-light.svg" width="200"/>
        <br>
        <a href="https://docs.arize.com/phoenix/">Docs</a>
        |
        <a href="https://github.com/Arize-ai/phoenix">GitHub</a>
        |
        <a href="https://join.slack.com/t/arize-ai/shared_invite/zt-1px8dcmlf-fmThhDFD_V_48oU7ALan4Q">Community</a>
    </p>
</center>
<h1 align="center">Arize-Phoenix LlamaPack</h1>

This LlamaPack instruments your LlamaIndex app for LLM tracing with [Phoenix](https://github.com/Arize-ai/phoenix), an open-source LLM observability library from [Arize AI](https://phoenix.arize.com/).

## Install and Import Dependencies
"""
logger.info("## Install and Import Dependencies")

# %pip install llama-index
# %pip install llama-index-readers-web
# %pip install llama-index-packs-arize-phoenix-query-engine



"""
Configure your MLX API key.
"""
logger.info("Configure your MLX API key.")

# os.environ["OPENAI_API_KEY"] = "copy-your-openai-api-key-here"

"""
Parse your documents into a list of nodes and pass to your LlamaPack. In this example, use nodes from a Paul Graham essay as input.
"""
logger.info("Parse your documents into a list of nodes and pass to your LlamaPack. In this example, use nodes from a Paul Graham essay as input.")

documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/jerryjliu/llama_index/adb054429f642cc7bbfcb66d4c232e072325eeab/examples/paul_graham_essay/data/paul_graham_essay.txt"
    ]
)
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
phoenix_pack = ArizePhoenixQueryEnginePack(nodes=nodes)

"""
Run a set of queries via the pack's `run` method, which delegates to the underlying query engine.
"""
logger.info("Run a set of queries via the pack's `run` method, which delegates to the underlying query engine.")

queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in tqdm(queries):
    logger.debug("Query")
    logger.debug("=====")
    logger.debug(query)
    logger.debug()
    response = phoenix_pack.run(query)
    logger.debug("Response")
    logger.debug("========")
    logger.debug(response)
    logger.debug()

"""
View your trace data in the Phoenix UI.
"""
logger.info("View your trace data in the Phoenix UI.")

phoenix_session_url = phoenix_pack.get_modules()["session_url"]
logger.debug(f"Open the Phoenix UI to view your trace data: {phoenix_session_url}")

"""
You can access the internals of the LlamaPack, including your Phoenix session and your query engine, via the `get_modules` method.
"""
logger.info("You can access the internals of the LlamaPack, including your Phoenix session and your query engine, via the `get_modules` method.")

phoenix_pack.get_modules()

"""
Check out the [Phoenix documentation](https://docs.arize.com/phoenix/) for more information!
"""
logger.info("Check out the [Phoenix documentation](https://docs.arize.com/phoenix/) for more information!")

logger.info("\n\n[DONE]", bright=True)