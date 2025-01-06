from pydantic import Field
from llama_index.core.llms.llm import LLM
from typing import Dict, Any
from llama_index.core.query_pipeline import (
    CustomQueryComponent,
    InputKeys,
    OutputKeys,
)
from llama_index.core.query_pipeline import InputComponent
from pyvis.network import Network
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import os
import llama_index.core
import phoenix as px
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# An Introduction to LlamaIndex Query Pipelines
#
# Overview
# LlamaIndex provides a declarative query API that allows you to chain together different modules in order to orchestrate simple-to-advanced workflows over your data.
#
# This is centered around our `QueryPipeline` abstraction. Load in a variety of modules (from LLMs to prompts to retrievers to other pipelines), connect them all together into a sequential chain or DAG, and run it end2end.
#
# **NOTE**: You can orchestrate all these workflows without the declarative pipeline abstraction (by using the modules imperatively and writing your own functions). So what are the advantages of `QueryPipeline`?
#
# - Express common workflows with fewer lines of code/boilerplate
# - Greater readability
# - Greater parity / better integration points with common low-code / no-code solutions (e.g. LangFlow)
# - [In the future] A declarative interface allows easy serializability of pipeline components, providing portability of pipelines/easier deployment to different systems.
#
# Cookbook
#
# In this cookbook we give you an introduction to our `QueryPipeline` interface and show you some basic workflows you can tackle.
#
# - Chain together prompt and LLM
# - Chain together query rewriting (prompt + LLM) with retrieval
# - Chain together a full RAG query pipeline (query rewriting, retrieval, reranking, response synthesis)
# - Setting up a custom query component
# - Executing a pipeline step-by-step

# Setup
#
# Here we setup some data + indexes (from PG's essay) that we'll be using in the rest of the cookbook.

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-postprocessor-cohere-rerank
# %pip install llama-index-llms-ollama


px.launch_app()

llama_index.core.set_global_handler("arize_phoenix")


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.llm = Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


reader = SimpleDirectoryReader(
    "./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries")

docs = reader.load_data()


if not os.path.exists("storage"):
    index = VectorStoreIndex.from_documents(docs)
    index.set_index_id("vector_index")
    index.storage_context.persist("./storage")
else:
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context, index_id="vector_index")

# 1. Chain Together Prompt and LLM
#
# In this section we show a super simple workflow of chaining together a prompt with LLM.
#
# We simply define `chain` on initialization. This is a special case of a query pipeline where the components are purely sequential, and we automatically convert outputs into the right format for the next inputs.


prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)
llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)

output = p.run(movie_name="The Departed")

print(str(output))

# View Intermediate Inputs/Outputs
#
# For debugging and other purposes, we can also view the inputs and outputs at each step.

output, intermediates = p.run_with_intermediates(movie_name="The Departed")

intermediates["8dc57d24-9691-4d8d-87d7-151865a7cd1b"]

intermediates["7ed9e26c-a704-4b0b-9cfd-991266e754c0"]

# Try Output Parsing
#
# Let's parse the outputs into a structured Pydantic object.


class Movie(BaseModel):
    """Object representing a single movie."""

    name: str = Field(..., description="Name of the movie.")
    year: int = Field(..., description="Year of the movie.")


class Movies(BaseModel):
    """Object representing a list of movies."""

    movies: List[Movie] = Field(..., description="List of movies.")


llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
output_parser = PydanticOutputParser(Movies)
json_prompt_str = """\
Please generate related movies to {movie_name}. Output with the following JSON format: 
"""
json_prompt_str = output_parser.format(json_prompt_str)

json_prompt_tmpl = PromptTemplate(json_prompt_str)

p = QueryPipeline(chain=[json_prompt_tmpl, llm, output_parser], verbose=True)
output = p.run(movie_name="Toy Story")

output

# Streaming Support
#
# The query pipelines have LLM streaming support (simply do `as_query_component(streaming=True)`). Intermediate outputs will get autoconverted, and the final output can be a streaming output. Here's some examples.

# **1. Chain multiple Prompts with Streaming**

prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)
prompt_str2 = """\
Here's some text:

{text}

Can you rewrite this with a summary of each movie?
"""
prompt_tmpl2 = PromptTemplate(prompt_str2)
llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
llm_c = llm.as_query_component(streaming=True)

p = QueryPipeline(
    chain=[prompt_tmpl, llm_c, prompt_tmpl2, llm_c], verbose=True
)

output = p.run(movie_name="The Dark Knight")
for o in output:
    print(o.delta, end="")

# **2. Feed streaming output to output parser**

p = QueryPipeline(
    chain=[
        json_prompt_tmpl,
        llm.as_query_component(streaming=True),
        output_parser,
    ],
    verbose=True,
)
output = p.run(movie_name="Toy Story")
print(output)

# Chain Together Query Rewriting Workflow (prompts + LLM) with Retrieval
#
# Here we try a slightly more complex workflow where we send the input through two prompts before initiating retrieval.
#
# 1. Generate question about given topic.
# 2. Hallucinate answer given question, for better retrieval.
#
# Since each prompt only takes in one input, note that the `QueryPipeline` will automatically chain LLM outputs into the prompt and then into the LLM.
#
# You'll see how to define links more explicitly in the next section.


prompt_str1 = "Please generate a concise question about Paul Graham's life regarding the following topic {topic}"
prompt_tmpl1 = PromptTemplate(prompt_str1)
prompt_str2 = (
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{query_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)
prompt_tmpl2 = PromptTemplate(prompt_str2)

llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
retriever = index.as_retriever(similarity_top_k=5)
p = QueryPipeline(
    chain=[prompt_tmpl1, llm, prompt_tmpl2, llm, retriever], verbose=True
)

nodes = p.run(topic="college")
len(nodes)

# Create a Full RAG Pipeline as a DAG
#
# Here we chain together a full RAG pipeline consisting of query rewriting, retrieval, reranking, and response synthesis.
#
# Here we can't use `chain` syntax because certain modules depend on multiple inputs (for instance, response synthesis expects both the retrieved nodes and the original question). Instead we'll construct a DAG explicitly, through `add_modules` and then `add_link`.

# 1. RAG Pipeline with Query Rewriting
#
# We use an LLM to rewrite the query first before passing it to our downstream modules - retrieval/reranking/synthesis.


prompt_str = "Please generate a question about Paul Graham's life regarding the following topic {topic}"
prompt_tmpl = PromptTemplate(prompt_str)
llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
retriever = index.as_retriever(similarity_top_k=3)
reranker = CohereRerank()
summarizer = TreeSummarize(llm=llm)

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "llm": llm,
        "prompt_tmpl": prompt_tmpl,
        "retriever": retriever,
        "summarizer": summarizer,
        "reranker": reranker,
    }
)

# Next we draw links between modules with `add_link`. `add_link` takes in the source/destination module ids, and optionally the `source_key` and `dest_key`. Specify the `source_key` or `dest_key` if there are multiple outputs/inputs respectively.
#
# You can view the set of input/output keys for each module through `module.as_query_component().input_keys` and `module.as_query_component().output_keys`.
#
# Here we explicitly specify `dest_key` for the `reranker` and `summarizer` modules because they take in two inputs (query_str and nodes).

p.add_link("prompt_tmpl", "llm")
p.add_link("llm", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("llm", "reranker", dest_key="query_str")
p.add_link("reranker", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")

print(summarizer.as_query_component().input_keys)

# We use `networkx` to store the graph representation. This gives us an easy way to view the DAG!


net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(p.dag)
net.show("rag_dag.html")

response = p.run(topic="YC")

print(str(response))

response = await p.arun(topic="YC")
print(str(response))

# 2. RAG Pipeline without Query Rewriting
#
# Here we setup a RAG pipeline without the query rewriting step.
#
# Here we need a way to link the input query to both the retriever, reranker, and summarizer. We can do this by defining a special `InputComponent`, allowing us to link the inputs to multiple downstream modules.


retriever = index.as_retriever(similarity_top_k=5)
summarizer = TreeSummarize(llm=Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096))
reranker = CohereRerank()

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "retriever": retriever,
        "summarizer": summarizer,
    }
)
p.add_link("input", "retriever")
p.add_link("input", "summarizer", dest_key="query_str")
p.add_link("retriever", "summarizer", dest_key="nodes")

output = p.run(input="what did the author do in YC")

print(str(output))

# Defining a Custom Component in a Query Pipeline
#
# You can easily define a custom component. Simply subclass a `QueryComponent`, implement validation/run functions + some helpers, and plug it in.
#
# Let's wrap the related movie generation prompt+LLM chain from the first example into a custom component.


class RelatedMovieComponent(CustomQueryComponent):
    """Related movie component."""

    llm: LLM = Field(..., description="Ollama LLM")

    def _validate_component_inputs(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"movie"}

    @property
    def _output_keys(self) -> set:
        return {"output"}

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        prompt_str = "Please generate related movies to {movie_name}"
        prompt_tmpl = PromptTemplate(prompt_str)
        p = QueryPipeline(chain=[prompt_tmpl, llm])
        return {"output": p.run(movie_name=kwargs["movie"])}

# Let's try the custom component out! We'll also add a step to convert the output to Shakespeare.


llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
component = RelatedMovieComponent(llm=llm)

prompt_str = """\
Here's some text:

{text}

Can you rewrite this in the voice of Shakespeare?
"""
prompt_tmpl = PromptTemplate(prompt_str)

p = QueryPipeline(chain=[component, prompt_tmpl, llm], verbose=True)

output = p.run(movie="Love Actually")

print(str(output))

# Stepwise Execution of a Pipeline
#
# Executing a pipeline one step at a time is a great idea if you:
# - want to better debug the order of execution
# - log data in between each step
# - give feedback to a user as to what is being processed
# - and more!
#
# To execute a pipeline, you must create a `run_state`, and then loop through the exection. A basic example is below.


prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)
llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)

run_state = p.get_run_state(movie_name="The Departed")

next_module_keys = p.get_next_module_keys(run_state)

while True:
    for module_key in next_module_keys:
        module = run_state.module_dict[module_key]
        module_input = run_state.all_module_inputs[module_key]

        output_dict = module.run_component(**module_input)

        p.process_component_output(
            output_dict,
            module_key,
            run_state,
        )

    next_module_keys = p.get_next_module_keys(
        run_state,
    )

    if not next_module_keys:
        run_state.result_outputs[module_key] = output_dict
        break

print(run_state.result_outputs[module_key]["output"].message.content)

logger.info("\n\n[DONE]", bright=True)
