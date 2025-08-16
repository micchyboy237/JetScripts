from autogen import UserProxyAgent
from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Optional Dependencies

## Different LLMs

AutoGen installs Ollama package by default. To use LLMs by other providers, you can install the following packages:
"""
logger.info("# Optional Dependencies")

pip install autogen[gemini,anthropic,mistral,together,groq,cohere]

"""
Check out the [notebook](/docs/notebooks/autogen_uniformed_api_calling) and
[blogpost](/blog/2024/06/24/AltModels-Classes) for more details.

## LLM Caching

To use LLM caching with Redis, you need to install the Python package with
the option `redis`:
"""
logger.info("## LLM Caching")

pip install "autogen[redis]"

"""
See [LLM Caching](/docs/topics/llm-caching) for details.

## IPython Code Executor

To use the IPython code executor, you need to install the `jupyter-client`
and `ipykernel` packages:
"""
logger.info("## IPython Code Executor")

pip install "autogen[ipython]"

"""
To use the IPython code executor:
"""
logger.info("To use the IPython code executor:")


proxy = UserProxyAgent(name="proxy", code_execution_config={"executor": "ipython-embedded"})

"""
## blendsearch

`pyautogen<0.2` offers a cost-effective hyperparameter optimization technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) for tuning Large Language Models. Please install with the [blendsearch] option to use it.
"""
logger.info("## blendsearch")

pip install "autogen[blendsearch]<0.2"

"""
Checkout [Optimize for Code Generation](https://github.com/autogenhub/autogen/blob/main/notebook/oai_completion.ipynb) and [Optimize for Math](https://github.com/autogenhub/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb) for details.

## retrievechat

`autogen` supports retrieval-augmented generation tasks such as question answering and code generation with RAG agents. Please install with the [retrievechat] option to use it with ChromaDB.
"""
logger.info("## retrievechat")

pip install "autogen[retrievechat]"

"""
Alternatively `autogen` also supports PGVector and Qdrant which can be installed in place of ChromaDB, or alongside it.
"""
logger.info("Alternatively `autogen` also supports PGVector and Qdrant which can be installed in place of ChromaDB, or alongside it.")

pip install "autogen[retrievechat-pgvector]"

"""

"""

pip install "autogen[retrievechat-qdrant]"

"""
RetrieveChat can handle various types of documents. By default, it can process
plain text and PDF files, including formats such as 'txt', 'json', 'csv', 'tsv',
'md', 'html', 'htm', 'rtf', 'rst', 'jsonl', 'log', 'xml', 'yaml', 'yml' and 'pdf'.
If you install [unstructured](https://unstructured-io.github.io/unstructured/installation/full_installation.html)
(`pip install "unstructured[all-docs]"`), additional document types such as 'docx',
'doc', 'odt', 'pptx', 'ppt', 'xlsx', 'eml', 'msg', 'epub' will also be supported.

You can find a list of all supported document types by using `autogen.retrieve_utils.TEXT_FORMATS`.

Example notebooks:

[Automated Code Generation and Question Answering with Retrieval Augmented Agents](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_RetrieveChat.ipynb)

[Group Chat with Retrieval Augmented Generation (with 5 group member agents and 1 manager agent)](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb)

[Automated Code Generation and Question Answering with Qdrant based Retrieval Augmented Agents](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_RetrieveChat_qdrant.ipynb)

## Teachability

To use Teachability, please install AutoGen with the [teachable] option.
"""
logger.info("## Teachability")

pip install "autogen[teachable]"

"""
Example notebook: [Chatting with a teachable agent](/docs/notebooks/agentchat_teachability)

## Large Multimodal Model (LMM) Agents

We offered Multimodal Conversable Agent and LLaVA Agent. Please install with the [lmm] option to use it.
"""
logger.info("## Large Multimodal Model (LMM) Agents")

pip install "autogen[lmm]"

"""
Example notebook: [LLaVA Agent](/docs/notebooks/agentchat_lmm_llava)

## mathchat

`pyautogen<0.2` offers an experimental agent for math problem solving. Please install with the [mathchat] option to use it.
"""
logger.info("## mathchat")

pip install "autogen[mathchat]<0.2"

"""
Example notebook: [Using MathChat to Solve Math Problems](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_MathChat.ipynb)

## Graph

To use a graph in `GroupChat`, particularly for graph visualization, please install AutoGen with the [graph] option.
"""
logger.info("## Graph")

pip install "autogen[graph]"

"""
Example notebook: [Finite State Machine graphs to set speaker transition constraints](/docs/notebooks/agentchat_groupchat_finite_state_machine)

## Long Context Handling

AutoGen includes support for handling long textual contexts by leveraging the LLMLingua library for text compression. To enable this functionality, please install AutoGen with the `[long-context]` option:
"""
logger.info("## Long Context Handling")

pip install "autogen[long-context]"

logger.info("\n\n[DONE]", bright=True)