from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import logger
from llama_index.core import Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer
import os
import shutil
import tiktoken


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
# Configuring Settings

The `Settings` is a bundle of commonly used resources used during the indexing and querying stage in a LlamaIndex workflow/application.

You can use it to set the global configuration. Local configurations (transformations, LLMs, embedding models) can be passed directly into the interfaces that make use of them.

The `Settings` is a simple singleton object that lives throughout your application. Whenever a particular component is not provided, the `Settings` object is used to provide it as a global default.

The following attributes can be configured on the `Settings` object:

## LLM

The LLM is used to respond to prompts and queries, and is responsible for writing natural language responses.
"""
logger.info("# Configuring Settings")


Settings.llm = Ollama(model="llama3.2", temperature=0.1)

"""
## Embed Model

The embedding model is used to convert text to numerical representations, used for calculating similarity and top-k retrieval.
"""
logger.info("## Embed Model")


Settings.embed_model = HuggingFaceEmbedding(
    model="nomic-embed-text", embed_batch_size=100
)

"""
## Node Parser / Text Splitter

The node parser / text splitter is used to parse documents into smaller chunks, called nodes.
"""
logger.info("## Node Parser / Text Splitter")


Settings.text_splitter = SentenceSplitter(chunk_size=1024)

"""
If you just want to change the chunk size or chunk overlap without changing the default splitter, this is also possible:
"""
logger.info("If you just want to change the chunk size or chunk overlap without changing the default splitter, this is also possible:")

Settings.chunk_size = 512
Settings.chunk_overlap = 20

"""
## Transformations

Transformations are applied to `Document`s during ingestion. By default, the `node_parser`/`text_splitter` is used, but this can be overridden and customized further.
"""
logger.info("## Transformations")


Settings.transformations = [SentenceSplitter(chunk_size=1024)]

"""
## Tokenizer

The tokenizer is used to count tokens. This should be set to something that matches the LLM you are using.
"""
logger.info("## Tokenizer")



Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode


Settings.tokenzier = AutoTokenizer.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)

"""
## Callbacks

You can set a global callback manager, which can be used to observe and consume events generated throughout the llama-index code
"""
logger.info("## Callbacks")


token_counter = TokenCountingHandler()
Settings.callback_manager = CallbackManager([token_counter])

"""
## Prompt Helper Arguments

A few specific arguments/values are used during querying, to ensure that the input prompts to the LLM have enough room to generate a certain number of tokens.

Typically these are automatically configured using attributes from the LLM, but they can be overridden in special cases.
"""
logger.info("## Prompt Helper Arguments")


Settings.context_window = 4096

Settings.num_output = 256

"""
<Aside type="tip">
Learn how to configure specific modules: - [LLM](/python/framework/module_guides/models/llms/usage_custom) - [Embedding Model](/python/framework/module_guides/models/embeddings) - [Node Parser/Text Splitters](/python/framework/module_guides/loading/node_parsers) - [Callbacks](/python/framework/module_guides/observability/callbacks)
</Aside>

## Setting local configurations

Interfaces that use specific parts of the settings can also accept local overrides.
"""
logger.info("## Setting local configurations")

index = VectorStoreIndex.from_documents(
    documents, embed_model=embed_model, transformations=transformations
)

query_engine = index.as_query_engine(llm=llm)

logger.info("\n\n[DONE]", bright=True)