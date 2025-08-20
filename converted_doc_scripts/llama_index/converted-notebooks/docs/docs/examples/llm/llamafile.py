from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llamafile import Llamafile
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
# llamafile

One of the simplest ways to run an LLM locally is using a [llamafile](https://github.com/Mozilla-Ocho/llamafile). llamafiles bundle model weights and a [specially-compiled](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#technical-details) version of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) into a single file that can run on most computers any additional dependencies. They also come with an embedded inference server that provides an [API](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints) for interacting with your model. 

## Setup

1) Download a llamafile from [HuggingFace](https://huggingface.co/models?other=llamafile)
2) Make the file executable
3) Run the file

Here's a simple bash script that shows all 3 setup steps:

```bash
# Download a llamafile from HuggingFace
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Make the file executable. On Windows, instead just rename the file to end in ".exe".
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Start the model server. Listens at http://localhost:8080 by default.
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser --embedding
```

Your model's inference server listens at localhost:8080 by default.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# llamafile")

# %pip install llama-index-llms-llamafile

# !pip install llama-index


llm = Llamafile(temperature=0, seed=0)

resp = llm.complete("Who is Octavia Butler?")

logger.debug(resp)

"""
**WARNING: TinyLlama's description of Octavia Butler above contains many falsehoods.** For example, she was born in California, not Pennsylvania. The information about her family and her education is a hallucation. She did not work as an elementary school teacher. Instead, she took a series of temporary jobs that would allow her to focus her energy on writing. Her work did not "quickly gain recognition": she sold her first short story around 1970, but did not gain prominence for another 14 years, when her short story "Speech Sounds" won the Hugo Award in 1984. Please refer to [Wikipedia](https://en.wikipedia.org/wiki/Octavia_E._Butler) for a real biography of Octavia Butler.

We use the TinyLlama model in this example notebook primarily because it's small and therefore quick to download for example purposes. A larger model might hallucinate less. However, this should serve as a reminder that LLMs often do lie, even about topics that are well-known enough to have a Wikipedia page. It's important verify their outputs with your own research.

#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system",
        content="Pretend you are a pirate with a colorful personality.",
    ),
    ChatMessage(role="user", content="What is your name?"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = llm.stream_complete("Who is Octavia Butler?")

for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system",
        content="Pretend you are a pirate with a colorful personality.",
    ),
    ChatMessage(role="user", content="What is your name?"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)