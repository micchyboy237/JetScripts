from IPython.display import display, Markdown
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Get data for RAG
"""
logger.info("## Get data for RAG")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'pg_essay.txt'

"""
## Build a simple RAG query engine 
- First we chunk documents from the main document, and
- Build in-memory Vector index from documents using LlamaIndex's `VectorStoreIndex` abstraction
"""
logger.info("## Build a simple RAG query engine")


documents = SimpleDirectoryReader(input_files=["pg_essay.txt"]).load_data()

index = VectorStoreIndex.from_documents(documents)

retriever = index.as_retriever(similarity_top_k=8)

query = "What did the author do during his time at Yale?"

nodes = retriever.retrieve(query)

nodes

"""
## Prompt compression using llmlingua2
LLMLingua2's claim to fame is its ability to achieve performant compression using a small prompt compression method trained via data distillation from GPT-4 for token classification! This performant compression comes with a performance bump of 3x-6x
> https://aclanthology.org/2024.findings-acl.57/
"""
logger.info("## Prompt compression using llmlingua2")

compressor_llmlingua2 = LongLLMLinguaPostprocessor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    device_map="mps",  # Mac users rejoice!
    use_llmlingua2=True,
)


results = compressor_llmlingua2._postprocess_nodes(
    nodes, query_bundle=QueryBundle(query_str=query)
)


display(Markdown(results[0].text))

query_engine1 = index.as_query_engine(
    similarity_top_k=8, postprocessors=[compressor_llmlingua2]
)

response = query_engine1.query(query)

display(Markdown(str(response)))

response.metadata

"""
## Test llmlingua 1
"""
logger.info("## Test llmlingua 1")

compressor_llmlingua1 = LongLLMLinguaPostprocessor(
    device_map="mps"  # Mac users rejoice!
)

results = compressor_llmlingua1._postprocess_nodes(
    nodes, query_bundle=QueryBundle(query_str=query)
)

results

query_engine_llmlingua1 = index.as_query_engine(
    similarity_top_k=8, postprocessors=[compressor_llmlingua1]
)

response = query_engine_llmlingua1.query(query)

display(Markdown(str(response)))

response.metadata

logger.info("\n\n[DONE]", bright=True)