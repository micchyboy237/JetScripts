from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-multi-tenancy-rag/examples/multi_tenancy_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Tenancy RAG

This notebook shows how to implement Multi-Tenancy RAG with MultiTenancyRAGPack.

### Setup
"""
logger.info("# Multi-Tenancy RAG")


# os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

"""
### Download data
"""
logger.info("### Download data")

# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2312.04511.pdf" -O "llm_compiler.pdf"
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2312.06648.pdf" -O "dense_x_retrieval.pdf"

"""
### Load Data
"""
logger.info("### Load Data")


reader = SimpleDirectoryReader(input_files=["dense_x_retrieval.pdf"])
dense_x_retrieval_docs = reader.load_data()

reader = SimpleDirectoryReader(input_files=["llm_compiler.pdf"])
llm_compiler_docs = reader.load_data()

"""
### Download `MultiTenancyRAGPack`
"""
logger.info("### Download `MultiTenancyRAGPack`")


MultiTenancyRAGPack = download_llama_pack(
    "MultiTenancyRAGPack", "./multitenancy_rag_pack"
)

multitenancy_rag_pack = MultiTenancyRAGPack()

"""
### Add documents for different users

Jerry -> Dense X Retrieval Paper

Ravi -> LLMCompiler Paper
"""
logger.info("### Add documents for different users")

multitenancy_rag_pack.add(documents=dense_x_retrieval_docs, user="Jerry")
multitenancy_rag_pack.add(documents=llm_compiler_docs, user="Ravi")

"""
### Querying for different users
"""
logger.info("### Querying for different users")

response = multitenancy_rag_pack.run(
    "what are propositions mentioned in the paper?", "Jerry"
)
logger.debug(response)

response = multitenancy_rag_pack.run("what are steps involved in LLMCompiler?", "Ravi")
logger.debug(response)

response = multitenancy_rag_pack.run("what are steps involved in LLMCompiler?", "Jerry")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)