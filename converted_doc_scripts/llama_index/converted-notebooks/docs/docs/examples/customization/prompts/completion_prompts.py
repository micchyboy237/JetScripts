from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/customization/prompts/completion_prompts.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Completion Prompts Customization

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Completion Prompts Customization")

# %pip install llama-index

"""
## Prompt Setup

Lets customize them to always answer, even if the context is not helpful!

Using `RichPromptTemplate`, we can define Jinja-formatted prompts.
"""
logger.info("## Prompt Setup")


text_qa_template_str = """Context information is below:
<context>
{{ context_str }}
</context>

Using both the context information and also using your own knowledge, answer the question:
{{ query_str }}
"""
text_qa_template = RichPromptTemplate(text_qa_template_str)

refine_template_str = """New context information has been provided:
<context>
{{ context_msg }}
</context>

We also have an existing answer generated using previous context:
<existing_answer>
{{ existing_answer }}
</existing_answer>

Using the new context, either update the existing answer, or repeat it if the new context is not relevant, when answering this query:
{query_str}
"""
refine_template = RichPromptTemplate(refine_template_str)

"""
## Using the Prompts

Now, we use the prompts in an index query!
"""
logger.info("## Using the Prompts")


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.llm = MLX(model="qwen3-1.7b-4bit-mini")
Settings.embed_model = MLXEmbedding(model_name="mxbai-embed-large")

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
### Before Adding Templates

Lets see the default existing prompts:
"""
logger.info("### Before Adding Templates")

query_engine.get_prompts()

"""
And how do they respond when asking about unrelated concepts?
"""
logger.info("And how do they respond when asking about unrelated concepts?")

logger.debug(query_engine.query("Who is Joe Biden?"))

"""
### After Adding Templates

Now, we can update the templates and observe the change in response!
"""
logger.info("### After Adding Templates")

query_engine.update_prompts(
    {
        "response_synthesizer:text_qa_template": text_qa_template,
        "response_synthesizer:refine_template": refine_template,
    }
)

logger.debug(query_engine.query("Who is Joe Biden?"))

logger.info("\n\n[DONE]", bright=True)