import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from semantic_kernel import Kernel
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.hugging_face import HuggingFacePromptExecutionSettings
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion, HuggingFaceTextEmbedding
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.prompt_template import PromptTemplateConfig
from services import Service
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using Hugging Face With Plugins

In this notebook, we demonstrate using Hugging Face models for Plugins using both SemanticMemory and text completions.

SK supports downloading models from the Hugging Face that can perform the following tasks: text-generation, text2text-generation, summarization, and sentence-similarity. You can search for models by task at https://huggingface.co/models.
"""
logger.info("# Using Hugging Face With Plugins")

# %pip install -U semantic-kernel

__version__


selectedService = Service.HuggingFace
logger.debug(f"Using service type: {selectedService}")

"""
First, we will create a kernel and add both text completion and embedding services.

For text completion, we are choosing GPT2. This is a text-generation model. (Note: text-generation will repeat the input in the output, text2text-generation will not.)
For embeddings, we are using sentence-transformers/all-MiniLM-L6-v2. Vectors generated for this model are of length 384 (compared to a length of 1536 from Ollama ADA).

The following step may take a few minutes when run for the first time as the models will be downloaded to your local machine.
"""
logger.info("First, we will create a kernel and add both text completion and embedding services.")


kernel = Kernel()

if selectedService == Service.HuggingFace:
    text_service_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    kernel.add_service(
        service=HuggingFaceTextCompletion(
            service_id=text_service_id, ai_model_id=text_service_id, task="text-generation"
        ),
    )
    embed_service_id = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_svc = HuggingFaceTextEmbedding(service_id=embed_service_id, ai_model_id=embed_service_id)
    kernel.add_service(
        service=embedding_svc,
    )
    memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_svc)
    kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

"""
### Add Memories and Define a plugin to use them

Most models available on huggingface.co are not as powerful as Ollama GPT-3+. Your plugins will likely need to be simpler to accommodate this.
"""
logger.info("### Add Memories and Define a plugin to use them")


collection_id = "generic"

async def run_async_code_0c12c14f():
    await memory.save_information(collection=collection_id, id="info1", text="Sharks are fish.")
    return 
 = asyncio.run(run_async_code_0c12c14f())
logger.success(format_json())
async def run_async_code_e9335008():
    await memory.save_information(collection=collection_id, id="info2", text="Whales are mammals.")
    return 
 = asyncio.run(run_async_code_e9335008())
logger.success(format_json())
async def run_async_code_467c036d():
    await memory.save_information(collection=collection_id, id="info3", text="Penguins are birds.")
    return 
 = asyncio.run(run_async_code_467c036d())
logger.success(format_json())
async def run_async_code_d9474fca():
    await memory.save_information(collection=collection_id, id="info4", text="Dolphins are mammals.")
    return 
 = asyncio.run(run_async_code_d9474fca())
logger.success(format_json())
async def run_async_code_e0b02293():
    await memory.save_information(collection=collection_id, id="info5", text="Flies are insects.")
    return 
 = asyncio.run(run_async_code_e0b02293())
logger.success(format_json())

my_prompt = """I know these animal facts:
- {{recall 'fact about sharks'}}
- {{recall 'fact about whales'}}
- {{recall 'fact about penguins'}}
- {{recall 'fact about dolphins'}}
- {{recall 'fact about flies'}}
Now, tell me something about: {{$request}}"""

execution_settings = HuggingFacePromptExecutionSettings(
    service_id=text_service_id,
    ai_model_id=text_service_id,
    max_tokens=45,
    temperature=0.5,
    top_p=0.5,
)

prompt_template_config = PromptTemplateConfig(
    template=my_prompt,
    name="text_complete",
    template_format="semantic-kernel",
    execution_settings=execution_settings,
)

my_function = kernel.add_function(
    function_name="text_complete",
    plugin_name="TextCompletionPlugin",
    prompt_template_config=prompt_template_config,
)

"""
Let's now see what the completion looks like! Remember, "gpt2" is nowhere near as large as ChatGPT, so expect a much simpler answer.
"""
logger.info("Let's now see what the completion looks like! Remember, "gpt2" is nowhere near as large as ChatGPT, so expect a much simpler answer.")

async def async_func_0():
    output = await kernel.invoke(
        my_function,
        request="What are whales?",
    )
    return output
output = asyncio.run(async_func_0())
logger.success(format_json(output))

output = str(output).strip()

async def async_func_7():
    query_result1 = await memory.search(
        collection=collection_id, query="What are sharks?", limit=1, min_relevance_score=0.3
    )
    return query_result1
query_result1 = asyncio.run(async_func_7())
logger.success(format_json(query_result1))

logger.debug(f"The queried result for 'What are sharks?' is {query_result1[0].text}")

logger.debug(f"{text_service_id} completed prompt with: '{output}'")

logger.info("\n\n[DONE]", bright=True)