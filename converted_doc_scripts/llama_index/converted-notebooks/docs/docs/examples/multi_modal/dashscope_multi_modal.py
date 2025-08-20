from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.dashscope import (
DashScopeMultiModal,
DashScopeMultiModalModels,
)
from llama_index.multi_modal_llms.dashscope.utils import (
create_dashscope_multi_modal_chat_message,
)
from llama_index.multi_modal_llms.dashscope.utils import load_local_images
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/dashscope_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal LLM using DashScope qwen-vl model for image reasoning

In this notebook, we show how to use DashScope qwen-vl MultiModal LLM class/abstraction for image understanding/reasoning.
Async is not currently supported

We also show several functions we are now supporting for DashScope LLM:
* `complete` (sync): for a single prompt and list of images
* `chat` (sync): for multiple chat messages
* `stream complete` (sync): for steaming output of complete
* `stream chat` (sync): for steaming output of chat
* multi round conversation.
"""
logger.info("# Multi-Modal LLM using DashScope qwen-vl model for image reasoning")

# !pip install -U llama-index-multi-modal-llms-dashscope

"""
##  Use DashScope to understand Images from URLs
"""
logger.info("##  Use DashScope to understand Images from URLs")

# %env DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY

"""
## Initialize `DashScopeMultiModal` and Load Images from URLs
"""
logger.info("## Initialize `DashScopeMultiModal` and Load Images from URLs")




image_urls = [
    "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
]

image_documents = load_image_urls(image_urls)

dashscope_multi_modal_llm = DashScopeMultiModal(
    model_name=DashScopeMultiModalModels.QWEN_VL_MAX,
)

"""
### Complete a prompt with images
"""
logger.info("### Complete a prompt with images")

complete_response = dashscope_multi_modal_llm.complete(
    prompt="What's in the image?",
    image_documents=image_documents,
)
logger.debug(complete_response)

multi_image_urls = [
    "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    "https://dashscope.oss-cn-beijing.aliyuncs.com/images/panda.jpeg",
]

multi_image_documents = load_image_urls(multi_image_urls)
complete_response = dashscope_multi_modal_llm.complete(
    prompt="What animals are in the pictures?",
    image_documents=multi_image_documents,
)
logger.debug(complete_response)

"""
### Steam Complete a prompt with a bunch of images
"""
logger.info("### Steam Complete a prompt with a bunch of images")

stream_complete_response = dashscope_multi_modal_llm.stream_complete(
    prompt="What's in the image?",
    image_documents=image_documents,
)

for r in stream_complete_response:
    logger.debug(r.delta, end="")

"""
### multi round conversation with chat messages
"""
logger.info("### multi round conversation with chat messages")


chat_message_user_1 = create_dashscope_multi_modal_chat_message(
    "What's in the image?", MessageRole.USER, image_documents
)
chat_response = dashscope_multi_modal_llm.chat([chat_message_user_1])
logger.debug(chat_response.message.content[0]["text"])
chat_message_assistent_1 = create_dashscope_multi_modal_chat_message(
    chat_response.message.content[0]["text"], MessageRole.ASSISTANT, None
)
chat_message_user_2 = create_dashscope_multi_modal_chat_message(
    "what are they doing?", MessageRole.USER, None
)
chat_response = dashscope_multi_modal_llm.chat(
    [chat_message_user_1, chat_message_assistent_1, chat_message_user_2]
)
logger.debug(chat_response.message.content[0]["text"])

"""
### Stream Chat through a list of chat messages
"""
logger.info("### Stream Chat through a list of chat messages")

stream_chat_response = dashscope_multi_modal_llm.stream_chat(
    [chat_message_user_1, chat_message_assistent_1, chat_message_user_2]
)
for r in stream_chat_response:
    logger.debug(r.delta, end="")

"""
###  Use images from local files
 Use local file:  
    Linux&mac file schema: file:///home/images/test.png  
    Windows file schema: file://D:/images/abc.png
"""
logger.info("###  Use images from local files")


local_images = [
    "file://THE_FILE_PATH1",
    "file://THE_FILE_PATH2",
]

image_documents = load_local_images(local_images)
chat_message_local = create_dashscope_multi_modal_chat_message(
    "What animals are in the pictures?", MessageRole.USER, image_documents
)
chat_response = dashscope_multi_modal_llm.chat([chat_message_local])
logger.debug(chat_response.message.content[0]["text"])

logger.info("\n\n[DONE]", bright=True)