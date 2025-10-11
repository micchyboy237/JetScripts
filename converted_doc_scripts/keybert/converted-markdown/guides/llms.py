from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from jet.adapters.keybert import KeyLLM
from keybert.llm import Cohere
from keybert.llm import LangChain
from keybert.llm import LiteLLM
from keybert.llm import Ollama
from keybert.llm import TextGeneration
from torch import cuda, bfloat16
import cohere
import ollama
import os
import shutil
import transformers


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# Large Language Models (LLM)
In this tutorial we will be going through the Large Language Models (LLM) that can be used in KeyLLM.
Having the option to choose the LLM allow you to leverage the model that suit your use-case.

### **Ollama**
To use Ollama's external API, we need to define our key and use the `keybert.llm.Ollama` model.

We install the package first:
"""
logger.info("# Large Language Models (LLM)")

pip install ollama

"""
Then we run Ollama as follows:
"""
logger.info("Then we run Ollama as follows:")


client = ollama.Ollama(api_key=MY_API_KEY)
llm = Ollama(client)

kw_model = KeyLLM(llm)

keywords = kw_model.extract_keywords(MY_DOCUMENTS)

"""
If you want to use a chat-based model, please run the following instead:
"""
logger.info("If you want to use a chat-based model, please run the following instead:")


client = ollama.Ollama(api_key=MY_API_KEY)
llm = Ollama(client, model="llama3.2", chat=True)

kw_model = KeyLLM(llm)

"""
### **Cohere**
To use Cohere's external API, we need to define our key and use the `keybert.llm.Cohere` model.

We install the package first:
"""
logger.info("### **Cohere**")

pip install cohere

"""
Then we run Cohere as follows:
"""
logger.info("Then we run Cohere as follows:")


co = cohere.Client(my_api_key)
llm = Cohere(co)

kw_model = KeyLLM(llm)

keywords = kw_model.extract_keywords(MY_DOCUMENTS)

"""
### **LiteLLM**
[LiteLLM](https://github.com/BerriAI/litellm) allows you to use any closed-source LLM with KeyLLM

We install the package first:
"""
logger.info("### **LiteLLM**")

pip install litellm

"""
Let's use Ollama as an example:
"""
logger.info("Let's use Ollama as an example:")


# os.environ["OPENAI_API_KEY"] = "sk-..."
llm = LiteLLM("gpt-3.5-turbo")

kw_model = KeyLLM(llm)

"""
### 🤗 **Hugging Face Transformers**
To use a Hugging Face transformers model, load in a pipeline and point
to any model found on their model hub (https://huggingface.co/models). Let's use Llama 2 as an example:
"""
logger.info("### 🤗 **Hugging Face Transformers**")


model_id = 'meta-llama/Llama-2-7b-chat-hf'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()

generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1
)

"""
Then, we load the `generator` in `KeyLLM` with a custom prompt:
"""
logger.info("Then, we load the `generator` in `KeyLLM` with a custom prompt:")


prompt = """
<s>[INST] <<SYS>>

You are a helpful assistant specialized in extracting comma-separated keywords.
You are to the point and only give the answer in isolation without any chat-based fluff.

<</SYS>>
I have the following document:
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST] meat, beef, eat, eating, emissions, steak, food, health, processed, chicken [INST]

I have the following document:
- [DOCUMENT]

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST]
"""

llm = TextGeneration(generator, prompt=prompt)
kw_model = KeyLLM(llm)

"""
### **LangChain**

To use `langchain` LLM client in KeyLLM, we can simply load in any LLM in `langchain` and pass that to KeyLLM.

We install langchain and corresponding LLM provider package first. Take Ollama as an example:
"""
logger.info("### **LangChain**")

pip install langchain
pip install langchain-ollama # LLM provider package

"""
> [!NOTE]
> KeyBERT only supports `langchain >= 0.1`


Then create your LLM client with `langchain`
"""
logger.info("Then create your LLM client with `langchain`")


_llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

"""
Finally, pass the `langchain` llm client to KeyBERT as follows:
"""
logger.info("Finally, pass the `langchain` llm client to KeyBERT as follows:")


llm = LangChain(_llm)

kw_model = KeyLLM(llm)

keywords = kw_model.extract_keywords(MY_DOCUMENTS)

logger.info("\n\n[DONE]", bright=True)