from jet.models.config import MODELS_CACHE_DIR
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts import RichPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/customization/prompts/chat_prompts.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chat Prompts Customization

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chat Prompts Customization")

# %pip install llama-index

"""
## Prompt Setup

Lets customize them to always answer, even if the context is not helpful!

Using `RichPromptTemplate`, we can define Jinja-formatted prompts.
"""
logger.info("## Prompt Setup")


chat_text_qa_prompt_str = """
{% chat role="system" %}
Always answer the question, even if the context isn't helpful.
{% endchat %}

{% chat role="user" %}
The following is some retrieved context:

<context>
{{ context_str }}
</context>

Using the context, answer the provided question:
{{ query_str }}
{% endchat %}
"""
text_qa_template = RichPromptTemplate(chat_text_qa_prompt_str)

chat_refine_prompt_str = """
{% chat role="system" %}
Always answer the question, even if the context isn't helpful.
{% endchat %}

{% chat role="user" %}
The following is some new retrieved context:

<context>
{{ context_msg }}
</context>

And here is an existing answer to the query:
<existing_answer>
{{ existing_answer }}
</existing_answer>

Using both the new retrieved context and the existing answer, either update or repeat the existing answer to this query:
{{ query_str }}
{% endchat %}
"""
refine_template = RichPromptTemplate(chat_refine_prompt_str)

"""
## Using the Prompts

Now, we use the prompts in an index query!
"""
logger.info("## Using the Prompts")


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."


Settings.llm = OllamaFunctionCalling(model="llama3.2")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

"""
### Before Customizing Templates

Lets see the default existing prompts:
"""
logger.info("### Before Customizing Templates")

query_engine.get_prompts()

"""
And how do they respond when asking about unrelated concepts?
"""
logger.info("And how do they respond when asking about unrelated concepts?")

logger.debug(query_engine.query("Who is Joe Biden?"))

"""
### After Customizing Templates

Now, we can update the templates and observe the change in response!
"""
logger.info("### After Customizing Templates")

query_engine.update_prompts(
    {
        "response_synthesizer:text_qa_template": text_qa_template,
        "response_synthesizer:refine_template": refine_template,
    }
)

logger.debug(query_engine.query("Who is Joe Biden?"))

logger.info("\n\n[DONE]", bright=True)