from jet.logger import logger
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.llms import LlamaCpp
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
ChatPromptTemplate,
HumanMessagePromptTemplate,
MessagesPlaceholder,
)
from langchain_experimental.chat_models import Llama2Chat
from os.path import expanduser
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
---
sidebar_label: Llama 2 Chat
---

# Llama2Chat

This notebook shows how to augment Llama-2 `LLM`s with the `Llama2Chat` wrapper to support the [Llama-2 chat prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). Several `LLM` implementations in LangChain can be used as interface to Llama-2 chat models. These include [ChatHuggingFace](/docs/integrations/chat/huggingface), [LlamaCpp](/docs/integrations/chat/llamacpp/), [GPT4All](/docs/integrations/llms/gpt4all), ..., to mention a few examples. 

`Llama2Chat` is a generic wrapper that implements `BaseChatModel` and can therefore be used in applications as [chat model](/docs/how_to#chat-models). `Llama2Chat` converts a list of Messages into the [required chat prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) and forwards the formatted prompt as `str` to the wrapped `LLM`.
"""
logger.info("# Llama2Chat")


"""
For the chat application examples below, we'll use the following chat `prompt_template`:
"""
logger.info("For the chat application examples below, we'll use the following chat `prompt_template`:")


template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

"""
## Chat with Llama-2 via `HuggingFaceTextGenInference` LLM

A HuggingFaceTextGenInference LLM encapsulates access to a [text-generation-inference](https://github.com/huggingface/text-generation-inference) server. In the following example, the inference server serves a [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) model. It can be started locally with:

```bash
docker run \
  --rm \
  --gpus all \
  --ipc=host \
  -p 8080:80 \
  -v ~/.cache/huggingface/hub:/data \
  -e HF_API_TOKEN=${HF_API_TOKEN} \
  ghcr.io/huggingface/text-generation-inference:0.9 \
  --hostname 0.0.0.0 \
  --model-id meta-llama/Llama-2-13b-chat-hf \
  --quantize bitsandbytes \
  --num-shard 4
```

This works on a machine with 4 x RTX 3080ti cards, for example. Adjust the `--num_shard` value to the number of GPUs available. The `HF_API_TOKEN` environment variable holds the Hugging Face API token.
"""
logger.info("## Chat with Llama-2 via `HuggingFaceTextGenInference` LLM")



"""
Create a `HuggingFaceTextGenInference` instance that connects to the local inference server and wrap it into `Llama2Chat`.
"""
logger.info("Create a `HuggingFaceTextGenInference` instance that connects to the local inference server and wrap it into `Llama2Chat`.")


llm = HuggingFaceTextGenInference(
    inference_server_url="http://127.0.0.1:8080/",
    max_new_tokens=512,
    top_k=50,
    temperature=0.1,
    repetition_penalty=1.03,
)

model = Llama2Chat(llm=llm)

"""
Then you are ready to use the chat `model` together with `prompt_template` and conversation `memory` in an `LLMChain`.
"""
logger.info("Then you are ready to use the chat `model` together with `prompt_template` and conversation `memory` in an `LLMChain`.")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

logger.debug(
    chain.run(
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
)

logger.debug(chain.run(text="Tell me more about #2."))

"""
## Chat with Llama-2 via `LlamaCPP` LLM

For using a Llama-2 chat model with a [LlamaCPP](/docs/integrations/llms/llamacpp) `LMM`, install the `llama-cpp-python` library using [these installation instructions](/docs/integrations/llms/llamacpp#installation). The following example uses a quantized [llama-2-7b-chat.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf) model stored locally at `~/Models/llama-2-7b-chat.Q4_0.gguf`. 

After creating a `LlamaCpp` instance, the `llm` is again wrapped into `Llama2Chat`
"""
logger.info("## Chat with Llama-2 via `LlamaCPP` LLM")



model_path = expanduser("~/Models/llama-2-7b-chat.Q4_0.gguf")

llm = LlamaCpp(
    model_path=model_path,
    streaming=False,
)
model = Llama2Chat(llm=llm)

"""
and used in the same way as in the previous example.
"""
logger.info("and used in the same way as in the previous example.")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

logger.debug(
    chain.run(
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
)

logger.debug(chain.run(text="Tell me more about #2."))

logger.info("\n\n[DONE]", bright=True)