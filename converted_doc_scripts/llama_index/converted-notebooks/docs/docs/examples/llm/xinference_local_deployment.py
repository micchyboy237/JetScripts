from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
TreeIndex,
VectorStoreIndex,
KeywordTableIndex,
KnowledgeGraphIndex,
SimpleDirectoryReader,
)
from llama_index.core import SummaryIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.xinference import Xinference
from xinference.client import RESTfulClient
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/xinference_local_deployment.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Xorbits Inference

In this demo notebook, we show how to use Xorbits Inference (Xinference for short) to deploy local LLMs in three steps.

We will be using the Llama 2 chat model in GGML format in the example, but the code should be easily transfrerable to all LLM chat models supported by Xinference. Below are a few examples:

| Name          | Type             | Language | Format  | Size (in billions) | Quantization                            |
|---------------|------------------|----------|---------|--------------------|-----------------------------------------|
| llama-2-chat  | RLHF Model       | en       | ggmlv3  | 7, 13, 70          | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| chatglm       | SFT Model        | en, zh   | ggmlv3  | 6                  | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |
| chatglm2      | SFT Model        | en, zh   | ggmlv3  | 6                  | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |
| wizardlm-v1.0 | SFT Model        | en       | ggmlv3  | 7, 13, 33          | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| wizardlm-v1.1 | SFT Model        | en       | ggmlv3  | 13                 | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| vicuna-v1.3   | SFT Model        | en       | ggmlv3  | 7, 13              | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |

The latest complete list of supported models can be found in Xorbits Inference's [official GitHub page](https://github.com/xorbitsai/inference/blob/main/README.md).

## <span style="font-size: xx-large;;">🤖  </span> Install Xinference

i. Run `pip install "xinference[all]"` in a terminal window.

ii. After installation is complete, restart this jupyter notebook.

iii. Run `xinference` in a new terminal window.

iv. You should see something similar to the following output:

```
INFO:xinference:Xinference successfully started. Endpoint: http://127.0.0.1:9997
INFO:xinference.core.service:Worker 127.0.0.1:21561 has been added successfully
INFO:xinference.deploy.worker:Xinference worker successfully started.
```

v. In the endpoint description, locate the endpoint port number after the colon. In the above case it is `9997`.

vi. Set the port number with the following cell:
"""
logger.info("# Xorbits Inference")

# %pip install llama-index-llms-xinference

port = 9997  # replace with your endpoint port number

"""
## <span style="font-size: xx-large;;">🚀  </span> Launch Local Models

In this step, we begin with importing the relevant libraries from `llama_index`

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("## <span style="font-size: xx-large;;">🚀  </span> Launch Local Models")

# !pip install llama-index


"""
Then, we launch a model and use it. This allows us to connect the model to documents and queries in later steps.

Feel free to change the parameters for better performance! In order to achieve optimal results, it is recommended to use models above 13B in size. That being said, 7B models is more than enough for this short demo.

Here are some more parameter options for the Llama 2 chat model in GGML format, listed from the least space-consuming to the most resource-intensive but high-performing. 


<span style="font-weight: bold; ;">model_size_in_billions:</span> 

`7`, `13`, `70`

<span style="font-weight: bold; ;">quantization for 7B and 13B models:</span> 

`q2_K`, `q3_K_L`, `q3_K_M`, `q3_K_S`, `q4_0`, `q4_1`, `q4_K_M`, `q4_K_S`, `q5_0`, `q5_1`, `q5_K_M`, `q5_K_S`, `q6_K`, `q8_0`

<span style="font-weight: bold; ;">quantizations for 70B models:</span>

`q4_0`
"""
logger.info("Then, we launch a model and use it. This allows us to connect the model to documents and queries in later steps.")

client = RESTfulClient(f"http://localhost:{port}")

model_uid = client.launch_model(
    model_name="llama-2-chat",
    model_size_in_billions=7,
    model_format="ggmlv3",
    quantization="q2_K",
)

llm = Xinference(
    endpoint=f"http://localhost:{port}",
    model_uid=model_uid,
    temperature=0.0,
    max_tokens=512,
)

"""
## <span style="font-size: xx-large;;">🕺  </span> Index the Data... and Chat!

In this step, we combine the model and the data to create a query engine. The query engine can then be used as a chat bot, answering our queries based on the given data.

We will be using `VetorStoreIndex` since it is relatively fast. That being said, feel free to change the index for different experiences. Here are some available indexes already imported from the previous step:

`ListIndex`, `TreeIndex`, `VetorStoreIndex`, `KeywordTableIndex`, `KnowledgeGraphIndex`

To change index, simply replace `VetorStoreIndex` with another index in the following code. 

The latest complete list of all available indexes can be found in Llama Index's [official Docs](https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/modules.html)
"""
logger.info("## <span style="font-size: xx-large;;">🕺  </span> Index the Data... and Chat!")

documents = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

index = VectorStoreIndex.from_documents(documents=documents)

query_engine = index.as_query_engine(llm=llm)

"""
We can optionally set the temperature and the max answer length (in tokens) directly through the `Xinference` object before asking a question. This allows us to change parameters for different questions without rebuilding the query engine every time.

`temperature` is a number between 0 and 1 that controls the randomness of responses. Higher values increase creativity but may lead to off-topic replies. Setting to zero guarentees the same response every time.

`max_tokens` is an integer that sets an upper bound for the response length. Increase it if answers seem cut off, but be aware that too long a response may exceed the context window and cause errors.
"""
logger.info("We can optionally set the temperature and the max answer length (in tokens) directly through the `Xinference` object before asking a question. This allows us to change parameters for different questions without rebuilding the query engine every time.")

llm.__dict__.update({"temperature": 0.0})
llm.__dict__.update({"max_tokens": 2048})

question = "What did the author do after his time at Y Combinator?"

response = query_engine.query(question)
display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)