from google.colab import userdata
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.aimon_rerank import AIMonRerank
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
<a href="https://colab.research.google.com/drive/1N4agIVU1NTEHaO5mLPa-bGBNpY17AxsO#scrollTo=8n2pDnW7521g" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# AIMon Rerank

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙
"""
logger.info("# AIMon Rerank")

# %%capture
# !pip install llama-index
# !pip install llama-index-postprocessor-aimon-rerank


"""
An MLX and AIMon API key is required for this notebook. Import the AIMon and MLX API keys from Colab Secrets
"""
logger.info("An MLX and AIMon API key is required for this notebook. Import the AIMon and MLX API keys from Colab Secrets")



os.environ["AIMON_API_KEY"] = userdata.get("AIMON_API_KEY")
# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

"""
Download data
"""
logger.info("Download data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
Generate documents and build an index
"""
logger.info("Generate documents and build an index")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents=documents)

"""
Define a task definition for the AIMon reranker and instantiate an instance of the AIMonRerank class. The [task definition](https://docs.aimon.ai/retrieval#task-definition) serves as an explicit instruction to the system, defining what the reranking evaluation should focus on.
"""
logger.info("Define a task definition for the AIMon reranker and instantiate an instance of the AIMonRerank class. The [task definition](https://docs.aimon.ai/retrieval#task-definition) serves as an explicit instruction to the system, defining what the reranking evaluation should focus on.")


task_definition = "Your task is to assess the actions of an individual specified in the user query against the context documents supplied."

aimon_rerank = AIMonRerank(
    top_n=2,
    api_key=userdata.get("AIMON_API_KEY"),
    task_definition=task_definition,
)

"""
#### Directly retrieve top 2 most similar nodes (i.e., without using a reranker)
"""
logger.info("#### Directly retrieve top 2 most similar nodes (i.e., without using a reranker)")

query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What did Sam Altman do in this essay?")

pprint_response(response, show_source=True)

"""
#### Retrieve top 10 most relevant nodes, but then rerank with AIMon Reranker

<img src="https://raw.githubusercontent.com/devvratbhardwaj/images/refs/heads/main/AIMon_Reranker.svg" alt="Diagram depicting working of AIMon reranker"/>

Explanation of the reranking process:

The diagram illustrates how a reranker refines document retrieval for a more accurate response.

1. **Initial Retrieval (Vector DB)**:  
   - A query is sent to the vector database.  
   - The system retrieves the **top 10 most relevant records** based on similarity scores (`top_k = 10`).  

2. **Reranking with AIMon**:  
   - Instead of using only the highest-scoring records directly, these 10 records are reranked using the **AIMon Reranker**.  
   - The reranker evaluates the documents based on their actual relevance to the query, rather than just raw similarity scores.  
   - During this step, a **task definition** is applied, serving as an explicit instruction that defines what the reranking evaluation should focus on.  
   - This ensures that the selected records are not just statistically similar but also **contextually relevant** to the intended task.  

3. **Final Selection (`top_n = 2`)**:  
   - After reranking, the system selects the **top 2 most contextually relevant records** for response generation.  
   - The **task definition ensures** that these records align with the query’s intent, leading to a **more precise and informative response**.
"""
logger.info("#### Retrieve top 10 most relevant nodes, but then rerank with AIMon Reranker")

query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[aimon_rerank]
)
response = query_engine.query("What did Sam Altman do in this essay?")

pprint_response(response, show_source=True)

"""
#### Conclusion

The AIMon reranker, using task definition, shifted retrieval focus from general YC leadership changes to Sam Altman’s specific actions. Initially, high-similarity documents lacked his decision-making details. After reranking, lower-similarity but contextually relevant documents highlighted his reluctance and timeline, ensuring a more accurate, task-aligned response over purely similarity-based retrieval.
"""
logger.info("#### Conclusion")

logger.info("\n\n[DONE]", bright=True)